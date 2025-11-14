import numpy as np
from rdkit import Chem
from collections import defaultdict
from scipy.spatial import cKDTree
from heapq import heappush, heappop
import pandas as pd

# Compare residue type using same residue grouping logic
residue_group_map = {
    'ASP': 'ASP/GLU', 'GLU': 'ASP/GLU',
    'LYS': 'LYS/ARG', 'ARG': 'LYS/ARG',
    'HIS': 'HIS',
    'CYS': 'CYS',
    'ASN': 'ASN/GLN/SER/THR', 'GLN': 'ASN/GLN/SER/THR',
    'SER': 'ASN/GLN/SER/THR', 'THR': 'ASN/GLN/SER/THR',
    'GLY': 'GLY',
    'PRO': 'PRO',
    'PHE': 'PHE/TYR/TRP', 'TYR': 'PHE/TYR/TRP', 'TRP': 'PHE/TYR/TRP',
    'ALA': 'ALA/ILE/LEU/MET/VAL', 'ILE': 'ALA/ILE/LEU/MET/VAL',
    'LEU': 'ALA/ILE/LEU/MET/VAL', 'MET': 'ALA/ILE/LEU/MET/VAL',
    'VAL': 'ALA/ILE/LEU/MET/VAL'
}

def create_grid(size=20, resolution=1):
    num_cells = int(size * resolution)
    grid = np.zeros((num_cells, num_cells, num_cells, 23))  # 23 features per grid point
    return grid

# Function to apply 3D rotation to atomic coordinates
def rotate_molecule(mol_to_rot, rotation_matrix):
    
    conf = mol_to_rot.GetConformer()
    for atom_idx in range(mol_to_rot.GetNumAtoms()):
        pos = conf.GetAtomPosition(atom_idx)
        new_pos = np.dot(rotation_matrix, np.array([pos.x, pos.y, pos.z]))
        conf.SetAtomPosition(atom_idx, new_pos)
    return mol_to_rot

# Generate a random rotation matrix
def generate_random_rotation_matrix():
    # Generate a random 3D rotation using Euler angles
    rotation = R.from_euler('xyz', np.random.uniform(0, 360, size=3), degrees=True)
    return rotation.as_matrix()

# Function to encode atomic features (same as before)
atom_types = {'C': 0, 'N': 1, 'O': 2, 'S': 3, 'other': 4}

def encode_atom_features(atom):
    features = np.zeros(14)
   # One-hot encoding for atom types
    atom_symbol = atom.GetSymbol()
    # if atom_symbol == 'H':
    #     print(atom_symbol, " is atom symbol")
    if atom_symbol in atom_types:
        features[atom_types[atom_symbol]] = 1
    else:
        features[atom_types['other']] = 1
    
    hybridization = atom.GetHybridization()
    if hybridization == Chem.HybridizationType.SP:
        features[5] = 1
    elif hybridization == Chem.HybridizationType.SP2:
        features[6] = 1
    elif hybridization == Chem.HybridizationType.SP3:
        features[7] = 1

    num_heavy_atoms = sum(1 for neighbor in atom.GetNeighbors() if neighbor.GetAtomicNum() > 1)
    features[8] = num_heavy_atoms
   
    # Number of bonded hetero atoms (atoms other than carbon and hydrogen)
    num_hetero_atoms = sum(1 for neighbor in atom.GetNeighbors() if neighbor.GetAtomicNum() not in {1, 6})
    features[9] = num_hetero_atoms
    features[10] = 1 if atom.GetIsAromatic() else 0

    # formal charge, 0 is no charge, 1 is negative, and 2 is positive charge
    residue = atom.GetPDBResidueInfo().GetResidueName().strip() 
    atom_name = atom.GetPDBResidueInfo().GetName().strip()

    if atom.GetFormalCharge() == 1: # setting positive charge to 2
        atom.SetFormalCharge(2) 

    if residue == "ASP" and atom_name == "CG": 
        atom.SetFormalCharge(1) # setting to negative charge, CG has summed up charge of OD1 and OD2
    if residue == "GLU" and atom_name == "CD": 
        atom.SetFormalCharge(1) # setting to negative charge, CD has summed up charge of OE1 and OE2

    features[11] = 1 if atom.GetFormalCharge() != 0 else 0 # binary label, charge or no charge
   
    features[12] = atom.GetFormalCharge()
    # if atom.GetFormalCharge() != 0:
    #     print(atom.GetFormalCharge(), "is atom that has formal charge and", atom_name, "is atom name")
    
    features[13] = 1 if atom.IsInRing() else 0
    
    #print(features)
    return features

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

# Map atoms to the grid based on their 3D coordinates
def map_atoms_to_grid(mol, grid, grid_center, grid_size=20, resolution=1):
    conf = mol.GetConformer()

    all_positions = np.array([[pos.x, pos.y, pos.z] for pos in [conf.GetAtomPosition(atom_idx) for atom_idx in range(mol.GetNumAtoms())]])
    min_coords = np.min(all_positions, axis=0)
    max_coords = np.max(all_positions, axis=0)
    scale = max_coords - min_coords

    def shift(pos, min_coords):
        return ((pos - min_coords))
    
    grid_to_atom_map = {}
    
    for atom in mol.GetAtoms(): 
        pos = conf.GetAtomPosition(atom.GetIdx())
        shifted_pos = shift(np.array([pos.x, pos.y, pos.z]), min_coords)

        # Map to grid coordinates
        grid_coord = np.rint(shifted_pos).astype(int)
        
        if np.all(grid_coord >= 0) and np.all(grid_coord < (grid_size * resolution)):
            atom_features = encode_atom_features(atom)

            residue = atom.GetPDBResidueInfo().GetResidueName()
            residue_features = encode_residue_type(residue)

            combined_features = np.concatenate((atom_features, residue_features))
            if np.any(grid[tuple(grid_coord)]):
                grid_coord = np.floor(shifted_pos).astype(int) # try flooring if rint doesn't work
                if np.any(grid[tuple(grid_coord)]):
                    grid_coord = np.ceil(shifted_pos).astype(int) # last ditch effort is to try ceiling if flooring fails
                    if np.any(grid[tuple(grid_coord)]):
                        print("Overwritten atoms")
                        raise Exception("Overwritten atoms!")
            grid[tuple(grid_coord)] = combined_features # print this part as well
        else:
            print("Atom didn't go in the grid")
            raise Exception("Atom out of bounds")
        
        grid_to_atom_map[tuple(grid_coord)] = {
            "atom_id": atom.GetIdx() + 1,
            "atom_type": atom.GetSymbol(),
            "residue": residue.strip()
        }

    return grid, grid_to_atom_map

def min_max_normalize(grid):
    min_val = np.min(grid)
    max_val = np.max(grid)
    
    if max_val - min_val == 0:
        return grid  # Avoid division by zero if all values are the same
    
    return (grid - min_val) / (max_val - min_val)

# Main function to generate multiple rotated grids
def generate_non_rotated_grid(grid_center, filtered_pdb_path, grid_size=30, resolution=1):
    mol = Chem.MolFromPDBFile(filtered_pdb_path, sanitize=True)
    
    if mol is None:
        return None
        
    # Create a new grid
    grid = create_grid(size=grid_size, resolution=resolution)
    
    # Map rotated atoms to the grid
    grid, mapped_grid = map_atoms_to_grid(mol, grid, grid_center, grid_size, resolution)

    # Apply Min-Max normalization
    grid = min_max_normalize(grid)
    
    return grid, mapped_grid

def print_features(features):
    atom_types_list = ['Carbon', 'Nitrogen', 'Oxygen', 'Sulfur', 'Other']
    residue_types_list = [
        'Aspartic Acid/Glutamic Acid',   # [0]
        'Lysine/Arginine',               # [1]
        'Histidine',                     # [2]
        'Cysteine',                      # [3]
        'Asparagine/Glutamine/Serine/Threonine',  # [4]
        'Glycine',                       # [5]
        'Proline',                       # [6]
        'Phenylalanine/Tyrosine/Tryptophan',  # [7]
        'Alanine/Isoleucine/Leucine/Methionine/Valine'  # [8]
    ]
    
    # === Atom Type ===
    atom_type_idx = np.argmax(features[:5])
    atom_type = atom_types_list[atom_type_idx]

    # === Hybridization ===
    hybridization = "Unknown"
    if features[5]:
        hybridization = "SP"
    elif features[6]:
        hybridization = "SP2"
    elif features[7]:
        hybridization = "SP3"

    num_heavy_atoms = features[8]
    num_hetero_atoms = features[9]
    is_aromatic = bool(features[10])
    has_charge = bool(features[11])
    formal_charge = features[12]
    in_ring = bool(features[13])

    # === Residue Type ===
    residue_type_idx = np.argmax(features[14:])
    residue_type = "Unknown"
    if features[14 + residue_type_idx] != 0:
        residue_type = residue_types_list[residue_type_idx]

    # === Print Information ===
    print(f"Atom: {atom_type}")
    print(f"Hybridization: {hybridization}")
    print(f"Number of heavy atoms bonded: {int(num_heavy_atoms)}")
    print(f"Number of hetero atoms bonded: {int(num_hetero_atoms)}")
    print(f"Aromatic: {'Yes' if is_aromatic else 'No'}")
    print(f"Has Charge: {'Yes' if has_charge else 'No'}")
    print(f"Formal Charge: {int(formal_charge)}")
    print(f"In Ring: {'Yes' if in_ring else 'No'}")
    print(f"Residue Type: {residue_type}\n")

def get_atom_and_residue_type(features):
    atom_types_list = ['C', 'N', 'O', 'S', 'other']
    residue_types_list = [
        'ASP/GLU', 'LYS/ARG', 'HIS', 'CYS',
        'ASN/GLN/SER/THR', 'GLY', 'PRO', 'PHE/TYR/TRP', 'ALA/ILE/LEU/MET/VAL'
    ]
    
    atom_idx = np.argmax(features[:5])
    atom_type = atom_types_list[atom_idx]
    
    residue_idx = np.argmax(features[14:])
    residue_type = residue_types_list[residue_idx] if features[14 + residue_idx] != 0 else 'Unknown'
    
    return atom_type, residue_type

def match_atoms_to_pdb(grid, file, df):
    grid_center = np.array([0, 0, 0])  # Grid center at origin

    non_rotated_grid, mapped_grid = generate_non_rotated_grid(grid_center, file)

    # Compute sum of absolute feature values at each (x, y, z)
    non_zero_mask = np.sum(np.abs(non_rotated_grid), axis=3) != 0  # Shape: (x, y, z)

    # Find coordinates where at least one feature is non-zero
    non_zero_coords = np.argwhere(non_zero_mask)

    number_of_atom_type_matches = 0
    number_of_residue_type_matches = 0

    for coord in non_zero_coords:
        x, y, z = coord
        features = non_rotated_grid[x, y, z, :]
        grid_atom_type, grid_residue_type = get_atom_and_residue_type(features)

        atom_info = mapped_grid.get((x, y, z))
        if not atom_info:
            print(f"WARNING: Grid coordinate ({x},{y},{z}) has no mapped atom!")
            continue

        # Compare atom type
        atom_type_match = (atom_info["atom_type"] == grid_atom_type)

        residue_category = residue_group_map.get(atom_info["residue"], 'Unknown')
        residue_match = (residue_category == grid_residue_type)

        if atom_type_match:
            number_of_atom_type_matches += 1
        if residue_match:
            number_of_residue_type_matches += 1

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

    for coord_rot in non_zero_coords_rot:
        feature_key = tuple(np.round(grid[tuple(coord_rot)], 5))
        candidate_coords = feature_to_coord_non_rot.get(feature_key, [])

        dists_rot, idxs_rot = tree_rot.query(coord_rot, k=10)
        neighbors_rot = [tuple(non_zero_coords_rot[i]) for i in idxs_rot if not np.array_equal(non_zero_coords_rot[i], coord_rot)]
        dists_rot = [d for i, d in zip(idxs_rot, dists_rot) if not np.array_equal(non_zero_coords_rot[i], coord_rot)]

        for coord_non_rot in candidate_coords:
            dists_non_rot, idxs_non_rot = tree_non_rot.query(coord_non_rot, k=10)
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

            if score >= 0.5:
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

    # Loop through rows
    for index, row in df.iterrows():
        x = int(row['x'])
        y = int(row['y'])
        z = int(row['z'])
        score = row['score']

        grid_features = grid[x, y, z, :]
        grid_atom_type, grid_residue_type = get_atom_and_residue_type(grid_features)

        mapped_coord = mapped_non_rot_coords.get((x, y, z))
        mapped_x, mapped_y, mapped_z = mapped_coord    
        atom_info = mapped_grid.get((mapped_x, mapped_y, mapped_z))
        mapped_atoms_info.append(atom_info)

        # Compare atom type
        atom_type_match = (atom_info["atom_type"] == grid_atom_type)

        residue_category = residue_group_map.get(atom_info["residue"], 'Unknown')
        residue_match = (residue_category == grid_residue_type)

        if atom_type_match:
            number_of_atom_type_matches += 1
        if residue_match:
            number_of_residue_type_matches += 1
        
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
