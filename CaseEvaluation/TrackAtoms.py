import numpy as np
from rdkit import Chem
from collections import defaultdict
from scipy.spatial import cKDTree
from heapq import heappush, heappop
import pandas as pd
import MDAnalysis as mda
from scipy.spatial.transform import Rotation as R

def create_grid(size=20, resolution=1):
    num_cells = int(size * resolution)
    grid = np.zeros((num_cells, num_cells, num_cells, 37))  # 23 features per grid point
    return grid

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

def one_hot_to_subtype(vec: np.ndarray) -> tuple[str, int]:
    """
    Convert a 37-d one-hot vector to (subtype_str, index).
    If multiple/non-binary entries exist, uses argmax. If all zeros, returns ('UNKNOWN', ATOM_INDEX['UNKNOWN']).
    """
    vec = np.asarray(vec).ravel()
    if vec.size != ATOM_ONEHOT_DIM:
        raise ValueError(f"Expected one-hot length {ATOM_ONEHOT_DIM}, got {vec.size}")
    idx = int(np.argmax(vec))
    # handle all-zero explicitly
    if vec[idx] <= 0:
        idx = ATOM_INDEX['UNKNOWN']
    return BIGGEST_SET[idx], idx

# Map atoms to the grid based on their 3D coordinates
def map_atoms_to_grid(df, grid, grid_center, grid_size=20, resolution=1):
    # Compute bounds for min max normalization
    coords = df[['X','Y','Z']].to_numpy(dtype=float)
    min_coords = np.min(coords, axis=0)
    shifted = coords - min_coords

    grid_to_atom_map = {}
    
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

        grid[target] = atom_feat

        subtype_str, subtype_idx = one_hot_to_subtype(atom_feat)

        grid_to_atom_map[target] = {
            "atom_id": int(idx) + 1,      # or use original PDB serial if you have it
            "atom_subtype": subtype_str,   # e.g., 'CA', 'OD2', 'NZ', 'UNKNOWN', ...
            "subtype_index": subtype_idx   # optional, handy for debugging
        }

    return grid, grid_to_atom_map

# Main function to generate multiple rotated grids
def generate_non_rotated_grid(grid_center, filtered_pdb_path, grid_size=30, resolution=1):
    pdb_df = pdb_to_dataframe(filtered_pdb_path)

    # Create a new grid
    grid = create_grid(size=grid_size, resolution=resolution)
    
    # Generate a random rotation matrix
    rotation_matrix = generate_random_rotation_matrix()
    
    # Rotate the molecule
    rotated_pdb_df = rotate_dataframe(pdb_df, rotation_matrix)
    
    # Map rotated atoms to the grid
    grid = map_atoms_to_grid(rotated_pdb_df, grid, grid_center, grid_size, resolution)
    
    return grid

def _index_to_subtype(idx: int) -> str:
    if 0 <= idx < len(BIGGEST_SET):
        return BIGGEST_SET[idx]
    return 'UNKNOWN'

def print_features(features: np.ndarray):
    """
    Expect a 37-dim one-hot vector over atom subtypes (BIGGEST_SET order).
    Prints only the chosen atom subtype.
    """
    features = np.asarray(features).ravel()
    if features.size != ATOM_ONEHOT_DIM:
        raise ValueError(f"Expected feature length {ATOM_ONEHOT_DIM}, got {features.size}")

    hot_idxs = np.where(features > 0.5)[0]

    if len(hot_idxs) == 0:
        subtype = 'UNKNOWN'
    elif len(hot_idxs) == 1:
        subtype = _index_to_subtype(int(hot_idxs[0]))
    else:
        # Multiple hots, pick argmax but warn
        subtype = _index_to_subtype(int(np.argmax(features)))
        print("Warning: multiple hot indices detected; used argmax.")

    #print(f"Atom subtype: {subtype}\n")

def get_atom_subtype(features: np.ndarray):
    features = np.asarray(features).ravel()
    if features.size != ATOM_ONEHOT_DIM:
        raise ValueError(f"Expected feature length {ATOM_ONEHOT_DIM}, got {features.size}")

    hot_idxs = np.where(features > 0.5)[0]
    if len(hot_idxs) == 0:
        subtype = 'UNKNOWN'
    else:
        subtype = _index_to_subtype(int(np.argmax(features)))

    return subtype

def match_atoms_to_pdb(grid, file, df):
    grid_center = np.array([0, 0, 0])  # Grid center at origin

    non_rotated_grid, mapped_grid = generate_non_rotated_grid(grid_center, file)

    #print(f"Grid shape: {non_rotated_grid.shape}")  # Should be (x, y, z, features)

    # Compute sum of absolute feature values at each (x, y, z)
    non_zero_mask = np.sum(np.abs(non_rotated_grid), axis=3) != 0  # Shape: (x, y, z)

    # Find coordinates where at least one feature is non-zero
    non_zero_coords = np.argwhere(non_zero_mask)

    #print(f"Total non-zero grid points: {non_zero_coords.shape[0]}\n")

    number_of_atom_type_matches = 0

    for coord in non_zero_coords:
        x, y, z = coord
        features = non_rotated_grid[x, y, z, :]
        grid_atom_type = get_atom_subtype(features)

        atom_info = mapped_grid.get((x, y, z))
        if not atom_info:
            print(f"WARNING: Grid coordinate ({x},{y},{z}) has no mapped atom!")
            continue

        # Compare atom type
        atom_type_match = (atom_info["atom_subtype"] == grid_atom_type)

        if atom_type_match:
            number_of_atom_type_matches += 1

        # print(f"Grid ({x},{y},{z}) --> Atom ID {atom_info['atom_id']}, Type {atom_info['atom_subtype']}")
        # print(f"   Grid Feature Atom Type: {grid_atom_type}, Match: {atom_type_match}")

    # Final summary
    total_points = len(non_zero_coords)

    # Non-zero coordinates in rotated grid
    non_zero_coords_rot = np.argwhere(np.sum(np.abs(grid), axis=3) != 0)

    # Non-zero coordinates in non-rotated grid
    non_zero_coords_non_rot = np.argwhere(np.sum(np.abs(non_rotated_grid), axis=3) != 0)

    # Build feature lookup dictionaries
    features_rot_dict = {tuple(coord): grid[tuple(coord)][...] for coord in non_zero_coords_rot}
    features_non_rot_dict = {tuple(coord): non_rotated_grid[tuple(coord)][...] for coord in non_zero_coords_non_rot}

    # Build KD-Trees for fast neighbor lookup
    tree_rot = cKDTree(non_zero_coords_rot)
    tree_non_rot = cKDTree(non_zero_coords_non_rot)

    # Build feature to coords lookup for non-rotated grid
    feature_to_coord_non_rot = defaultdict(list)
    for coord in non_zero_coords_non_rot:
        feature_key = tuple(np.round(non_rotated_grid[tuple(coord)], 5))
        feature_to_coord_non_rot[feature_key].append(tuple(coord))

    mapped_non_rot_coords = {}

    match_candidates = []

    number_of_neighbors = 1 if total_points < 5 else (10 if total_points >= 10 else 5)

    for coord_rot in non_zero_coords_rot:
        feature_key = tuple(np.round(grid[tuple(coord_rot)], 5))
        candidate_coords = feature_to_coord_non_rot.get(feature_key, [])

        dists_rot, idxs_rot = tree_rot.query(coord_rot, k=number_of_neighbors)
        neighbors_rot = [tuple(non_zero_coords_rot[i]) for i in idxs_rot if not np.array_equal(non_zero_coords_rot[i], coord_rot)]
        dists_rot = [d for i, d in zip(idxs_rot, dists_rot) if not np.array_equal(non_zero_coords_rot[i], coord_rot)]

        for coord_non_rot in candidate_coords:
            dists_non_rot, idxs_non_rot = tree_non_rot.query(coord_non_rot, k=number_of_neighbors)
            neighbors_non_rot = [tuple(non_zero_coords_non_rot[i]) for i in idxs_non_rot if not np.array_equal(non_zero_coords_non_rot[i], coord_non_rot)]
            dists_non_rot = [d for i, d in zip(idxs_non_rot, dists_non_rot) if not np.array_equal(non_zero_coords_non_rot[i], coord_non_rot)]

            # Compare neighbor features with distance weighting
            score = 0
            for n_rot, dist_rot in zip(neighbors_rot, dists_rot):
                f_rot = np.round(features_rot_dict[n_rot], 5)

                for n_non_rot, dist_non_rot in zip(neighbors_non_rot, dists_non_rot):
                    f_non_rot = np.round(features_non_rot_dict[n_non_rot], 5)

                    if np.allclose(f_rot, f_non_rot, atol=1e-5):
                        weight = 1.0 / (1.0 + (dist_rot + dist_non_rot) / 2.0)
                        score += weight
                        break

            heappush(match_candidates, (-score, tuple(coord_rot), tuple(coord_non_rot)))

    # Finalize unique best matches
    mapped_non_rot_coords = {}
    used_rot = set()
    used_non_rot = set()
    number_of_confident_matches = 0

    while match_candidates:
        neg_score, coord_rot, coord_non_rot = heappop(match_candidates)
        score = -neg_score

        if coord_rot not in used_rot and coord_non_rot not in used_non_rot:
            mapped_non_rot_coords[coord_rot] = coord_non_rot
            used_rot.add(coord_rot)
            used_non_rot.add(coord_non_rot)
            number_of_confident_matches += 1

    number_of_atom_type_matches = 0
    number_of_residue_type_matches = 0

    mapped_atoms_info = []

    #print(number_of_confident_matches, "is number of confident matches")

    # Loop through rows
    for index, row in df.iterrows():
        x = int(row['x'])
        y = int(row['y'])
        z = int(row['z'])
        score = row['score']

        grid_features = grid[x, y, z, :]
        grid_atom_type = get_atom_subtype(grid_features)

        mapped_coord = mapped_non_rot_coords.get((x, y, z))
        mapped_x, mapped_y, mapped_z = mapped_coord    
        atom_info = mapped_grid.get((mapped_x, mapped_y, mapped_z))
        mapped_atoms_info.append(atom_info)

        # Compare atom type
        atom_type_match = (atom_info["atom_subtype"] == grid_atom_type)

        if atom_type_match:
            number_of_atom_type_matches += 1
        
        # print(f"Grid ({x},{y},{z}) --> Atom ID {atom_info['atom_id']}, Type {atom_info['atom_type']}, Residue {atom_info['residue']}")
        # print(f"   Grid Feature Atom Type: {grid_atom_type}, Match: {atom_type_match}")
        # print(f"   Grid Feature Residue Type: {grid_residue_type}, Match: {residue_match}")
        # print_features(grid_features)
        # print(score, "is grad cam score\n")

    # Final summary
    total_points = len(non_zero_coords)
    # print(f"\nSummary:")
    # print(f"Correct Atom Type Matches: {number_of_atom_type_matches}/{total_points}")
    # print(f"Correct Residue Type Matches: {number_of_residue_type_matches}/{total_points}")

    return mapped_atoms_info
