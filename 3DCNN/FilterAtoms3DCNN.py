import MDAnalysis as mda
import pandas as pd
from biopandas.pdb import PandasPdb
import os
import glob
import re
import math
import numpy as np
from rdkit import Chem
from scipy.spatial.transform import Rotation as R

def grid_list(atom_df):
    return list(zip(atom_df['x_coord'], atom_df['y_coord'], atom_df['z_coord']))

def filtering_proteins(atom_df, grid_list, radius=5.0):
    atom_coords = atom_df[['x_coord', 'y_coord', 'z_coord']].values
    filtered_atoms = set()

    for x, y, z in grid_list:
        distances_sq = (atom_coords[:, 0] - x)**2 + (atom_coords[:, 1] - y)**2 + (atom_coords[:, 2] - z)**2
        mask = distances_sq <= radius**2
        filtered_atoms.update(atom_df.index[mask])

    print(f"Total atoms within {radius} Å cutoff: {len(filtered_atoms)}")
    return atom_df.loc[list(filtered_atoms)]

def pdb_to_dataframe(pdb_file):
    """
    Load a PDB file using MDAnalysis and convert key atom information to a pandas DataFrame.
    """
    u = mda.Universe(pdb_file)
    
    # Extract atom-related data: atom name, residue name, residue ID, and chain ID
    atom_data = {
        'Atom Name': u.atoms.names,
        'Residue Name': u.atoms.resnames,
        'Residue ID': u.atoms.resids,
        'Chain ID': u.atoms.segids,
        'X': u.atoms.positions[:, 0],
        'Y': u.atoms.positions[:, 1],
        'Z': u.atoms.positions[:, 2],
    }
    
    # Create a pandas DataFrame from the atom data
    df = pd.DataFrame(atom_data)
    
    return df

def get_protein_atoms(protein_file):
    protein_pdb_df = PandasPdb().read_pdb(protein_file)
    protein_pdb_df.df.keys()
    protein = protein_pdb_df.df['ATOM']
    protein = protein[~protein['atom_name'].str.startswith('H')] # don't use hydrogen

    return protein


def natural_sort_key(s):
    """Function to sort strings in a natural alphanumeric order."""
    return [int(text) if text.isdigit() else text.lower() for text in re.split(r'(\d+)', s)]

def create_grid(size=20, resolution=1):
    num_cells = int(size * resolution)
    grid = np.zeros((num_cells, num_cells, num_cells, 37))  # 23 features per grid point
    return grid

# Function to apply 3D rotation to atomic coordinates
def rotate_dataframe(df, rotation_matrix, origin='centroid', inplace=False):
    """
    Rotate coordinates in a PDB DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        Must have columns 'X','Y','Z' (float Å).
    rotation_matrix : (3,3) ndarray
        Proper rotation matrix.
    origin : {'centroid','mean', array-like of shape (3,), None}
        Point about which to rotate. 'centroid' (same as 'mean') subtracts the
        mean of coordinates before rotating, then adds it back. If an array is
        given, rotate about that fixed point. If None, rotate about (0,0,0).
    inplace : bool
        If True, updates df in place and returns df. Otherwise returns a copy.

    Returns
    -------
    pd.DataFrame
    """
    if not {'X','Y','Z'}.issubset(df.columns):
        raise ValueError("DataFrame must contain columns: 'X','Y','Z'.")

    # Choose working frame
    out = df if inplace else df.copy()

    # Extract coordinates (N,3)
    coords = out[['X','Y','Z']].to_numpy(dtype=float)

    # Determine rotation origin
    if origin in ('centroid', 'mean'):
        pivot = coords.mean(axis=0, keepdims=True)  # (1,3)
    elif origin is None:
        pivot = np.zeros((1,3), dtype=float)
    else:
        pivot = np.asarray(origin, dtype=float).reshape(1,3)

    # Rotate about pivot: (coords - pivot) @ R^T + pivot
    rotated = (coords - pivot) @ rotation_matrix.T + pivot

    # Write back
    out[['X','Y','Z']] = rotated
    return out

# Generate a random rotation matrix
def generate_random_rotation_matrix():
    # Generate a random 3D rotation using Euler angles
    rotation = R.from_euler('xyz', np.random.uniform(0, 360, size=3), degrees=True)
    return rotation.as_matrix()

BIGGEST_SET = [
    # Carbon (C) subtypes
    'C', 'CA', 'CB', 'CD', 'CD1', 'CD2', 'CE', 'CE1', 'CE2', 'CE3',
    'CG', 'CG1', 'CG2', 'CH2', 'CZ', 'CZ2', 'CZ3',

    # Oxygen (O) subtypes
    'O', 'OH', 'OD1', 'OD2', 'OE1', 'OE2', 'OG', 'OG1',

    # Nitrogen (N) subtypes
    'N', 'NE', 'NE1', 'NE2', 'ND1', 'ND2', 'NZ', 'NH1', 'NH2',

    # Sulfur (S) subtypes
    'SD', 'SG'
]
BIGGEST_SET.append('UNKNOWN')  # index for unknown atom names
ATOM_INDEX = {atom: i for i, atom in enumerate(BIGGEST_SET)}
ATOM_ONEHOT_DIM = len(BIGGEST_SET)  # 37 with the list above

def atom_one_hot_from_name(atom_name: str) -> np.ndarray:
    vec = np.zeros(ATOM_ONEHOT_DIM, dtype=float)
    key = (atom_name or "").strip().upper()
    idx = ATOM_INDEX.get(key, ATOM_INDEX['UNKNOWN'])
    vec[idx] = 1.0
    if key not in ATOM_INDEX:
        print(atom_name, "went to unknown column")
    return vec

# Function to perform one-hot encoding for residue types
def encode_residue_type(residue):
    features = np.zeros(9)
    if residue in ['ASP', 'GLU']:
        features[0] = 1
    elif residue in ['LYS', 'ARG']:
        features[1] = 1
    elif residue == 'HIS':
        features[2] = 1
    elif residue == 'CYS':
        features[3] = 1
    elif residue in ['ASN', 'GLN', 'SER', 'THR']:
        features[4] = 1
    elif residue == 'GLY':
        features[5] = 1
    elif residue == 'PRO':
        features[6] = 1
    elif residue in ['PHE', 'TYR', 'TRP']:
        features[7] = 1
    elif residue in ['ALA', 'ILE', 'LEU', 'MET', 'VAL']:
        features[8] = 1
    return features

def find_nearest_empty(grid: np.ndarray, gc: np.ndarray, G: int, max_radius: int = None):
    """
    Find the nearest empty voxel to gc by expanding L∞ shells.
    Returns a tuple (x,y,z) or None if none found within max_radius.
    """
    x0, y0, z0 = map(int, gc)
    if max_radius is None:
        max_radius = G  # worst-case fallback

    # If target is already empty, use it
    if 0 <= x0 < G and 0 <= y0 < G and 0 <= z0 < G and not np.any(grid[x0, y0, z0]):
        return (x0, y0, z0)

    for r in range(1, max_radius + 1):
        xmin, xmax = max(0, x0 - r), min(G - 1, x0 + r)
        ymin, ymax = max(0, y0 - r), min(G - 1, y0 + r)
        zmin, zmax = max(0, z0 - r), min(G - 1, z0 + r)

        best_cell = None
        best_d2 = np.inf

        # Scan only the shell (any coord on the boundary of the cube)
        for x in range(xmin, xmax + 1):
            for y in range(ymin, ymax + 1):
                for z in range(zmin, zmax + 1):
                    if not (x in (xmin, xmax) or y in (ymin, ymax) or z in (zmin, zmax)):
                        continue
                    if not np.any(grid[x, y, z]):
                        d2 = (x - x0) ** 2 + (y - y0) ** 2 + (z - z0) ** 2
                        if d2 < best_d2:
                            best_d2 = d2
                            best_cell = (x, y, z)

        if best_cell is not None:
            return best_cell

    return None

# Map atoms to the grid based on their 3D coordinates
def map_atoms_to_grid(df, grid, grid_center, grid_size=20, resolution=1):
    # Compute bounds for min max normalization
    coords = df[['X','Y','Z']].to_numpy(dtype=float)
    min_coords = np.min(coords, axis=0)
    shifted = coords - min_coords
    
    for idx, row in df.iterrows():
        spos = shifted[idx]
        gc = np.rint(spos).astype(int)

        # Try rint cell first
        target = None
        if 0 <= gc[0] < grid_size and 0 <= gc[1] < grid_size and 0 <= gc[2] < grid_size and not np.any(grid[tuple(gc)]):
            target = tuple(gc)
        else:
            # Find nearest empty voxel
            target = find_nearest_empty(grid, gc, grid_size, max_radius=2)

        if target is None:
            raise Exception(f"Atom at df index {idx} could not be placed (no empty voxel found).")
        
        atom_feat = atom_one_hot_from_name(row['Atom Name'])
        # res_feat  = encode_residue_type(row['Residue Name'])
        # combined  = np.concatenate([atom_feat, res_feat])  # length = expected_F

        grid[target] = atom_feat

    return grid

# Main function to generate multiple rotated grids
def generate_rotated_grids(grid_center, filtered_pdb_path, num_rotations=20, grid_size=30, resolution=1):
    pdb_df = pdb_to_dataframe(filtered_pdb_path)
    
    grids = []
    
    for i in range(num_rotations):
        # Create a new grid
        grid = create_grid(size=grid_size, resolution=resolution)
        
        # Generate a random rotation matrix
        rotation_matrix = generate_random_rotation_matrix()
        
        # Rotate the molecule
        rotated_pdb_df = rotate_dataframe(pdb_df, rotation_matrix)
        
        # Map rotated atoms to the grid
        grid = map_atoms_to_grid(rotated_pdb_df, grid, grid_center, grid_size, resolution)

        # Store the rotated grid
        grids.append(grid)
    
    return grids
def saving_features(rotated_grids,output_path,protein_name_):
    os.makedirs(output_path, exist_ok=True)
    # Save each grid
    for idx, grid in enumerate(rotated_grids):
        np.save(f'{output_path}/{protein_name_}_grid_{idx}.npy', grid)
        print(f"Saved rotated grid {idx} successfully.")
    return

def create_grids(protein_file, unlabeled_files, dataset_name):
    protein_pdb_df = PandasPdb().read_pdb(protein_file)
    protein_pdb_df.df.keys()
    protein = protein_pdb_df.df['ATOM']
    protein = protein[~protein['atom_name'].str.startswith('H')] # don't use hydrogen

    unlabeled_files = sorted(unlabeled_files, key=natural_sort_key)

    for unlabeled_file in unlabeled_files:
        fragment_df = PandasPdb().read_pdb(unlabeled_file)
        fragment_df.df.keys()
        fragment = fragment_df.df['HETATM']

        grid_list_ = grid_list(fragment)

        filtered_atoms = filtering_proteins(protein, grid_list_)
        
        if not filtered_atoms.empty:
            # Save to pdb
            filtered_pdb = PandasPdb()
            filtered_pdb.df['ATOM'] = filtered_atoms
            base_name = os.path.basename(unlabeled_file)

            # pat = re.compile(r"box(?P<box>\d+).*?mode[_-]?(?P<mode>\d+)\.pdb$", re.IGNORECASE)
            # m = pat.search(unlabeled_file)

            # if m:
            filtered_pdb_path = f"filtered-{dataset_name}-piezo-pdbs/unlabeled/{base_name[:7]}-filtered.pdb"
            os.makedirs(os.path.dirname(filtered_pdb_path), exist_ok=True)
            filtered_pdb.to_pdb(path=filtered_pdb_path, records=None, gz=False, append_newline=True)

    unlabeled_files = glob.glob(f"filtered-{dataset_name}-piezo-pdbs/unlabeled/*.pdb")
    unlabeled_files = sorted(unlabeled_files, key=natural_sort_key)

    for file in unlabeled_files:
        unlabeled_output_path = f"{dataset_name}-piezo-grids/unlabeled"

        grid_center = np.array([0, 0, 0])  # Grid center at origin
        
        rotated_grids = generate_rotated_grids(grid_center, file, num_rotations=1)
        base_name = os.path.splitext(os.path.basename(file))[0]
        saving_features(rotated_grids,unlabeled_output_path,base_name)

