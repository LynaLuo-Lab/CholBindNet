import MDAnalysis as mda
import pandas as pd
from biopandas.pdb import PandasPdb
import os
import glob
import re
import math
import numpy as np

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

def grid_list(atom_df):
    return list(zip(atom_df['x_coord'], atom_df['y_coord'], atom_df['z_coord']))

def filtering_proteins(atom_df, grid_list, radius=5.0):
    atom_coords = atom_df[['x_coord', 'y_coord', 'z_coord']].values
    filtered_atoms = set()

    for x, y, z in grid_list:
        distances_sq = (atom_coords[:, 0] - x)**2 + (atom_coords[:, 1] - y)**2 + (atom_coords[:, 2] - z)**2
        mask = distances_sq <= radius**2
        filtered_atoms.update(atom_df.index[mask])

    return atom_df.loc[list(filtered_atoms)]

def get_protein_name(filename):
    basename = os.path.basename(filename)  # Get file name without path
    match = re.match(r'([a-zA-Z0-9]{4})', basename)  # Match the first 4-character PDB ID
    if match:
        return match.group(1).upper()
    else:
        return None
def get_mode_index(filename):
    basename = os.path.basename(filename)
    match = re.search(r'mode_(\d+)', basename)
    if match:
        return int(match.group(1))
    else:
        return None  # or raise ValueError("No mode index found.")

def natural_sort_key(s):
    """Function to sort strings in a natural alphanumeric order."""
    return [int(text) if text.isdigit() else text.lower() for text in re.split(r'(\d+)', s)]

import numpy as np

def compute_inverse_pairwise_distances(df):
    """
    Compute the pairwise Euclidean distances between residues based on their 3D coordinates.

    Parameters:
    df (pd.DataFrame): DataFrame containing 'X', 'Y', 'Z' coordinates and 'NewIndex' as index.

    Returns:
    pd.DataFrame: A DataFrame containing the pairwise distance matrix.
    """
    # Extract the coordinates (X, Y, Z)
    coordinates = df[['X', 'Y', 'Z']].values

    # Calculate pairwise distances using broadcasting
    diff = coordinates[:, np.newaxis, :] - coordinates[np.newaxis, :, :]
    distances = np.sqrt(np.sum(diff ** 2, axis=-1))

    # Compute inverse distance (1/d)
    with np.errstate(divide='ignore'):  # Ignore division by zero warning
        inverse_distances = 1 / distances

    # Set diagonal elements (self-distances) to 1
    np.fill_diagonal(inverse_distances, 1)

    # Cap values at 1
    inverse_distances = np.minimum(inverse_distances, 1)

    return inverse_distances

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

def one_hot_encoding(pdb_df):
    biggest_set = [
        # Carbon (C) subtypes
        'C', 'CA', 'CB', 'CD', 'CD1', 'CD2', 'CE', 'CE1', 'CE2', 'CE3', 'CG', 'CG1', 'CG2', 'CH2', 'CZ', 'CZ2', 'CZ3',

        # Oxygen (O) subtypes
        'O', 'OH', 'OD1', 'OD2', 'OE1', 'OE2', 'OG', 'OG1', 

        # Nitrogen (N) subtypes
        'N', 'NE', 'NE1', 'NE2', 'ND1', 'ND2', 'NZ', 'NH1', 'NH2', 

        # Sulfur (S) subtypes
        'SD', 'SG'
    ]

    biggest_set.append('UNKNOWN')  # Add an additional column for unknown atom types
    
    # Create a zero matrix with shape (num_rows, num_unique_atoms)
    num_rows = len(pdb_df)
    num_cols = len(biggest_set)
    one_hot_matrix = np.zeros((num_rows, num_cols), dtype=int)

    # Create a mapping from atom name to index
    atom_to_index = {atom: idx for idx, atom in enumerate(biggest_set)}

    # Fill the one-hot matrix
    for i, atom in enumerate(pdb_df['Atom Name']):
        if atom in atom_to_index:
            one_hot_matrix[i, atom_to_index[atom]] = 1
        else:
            one_hot_matrix[i, atom_to_index['UNKNOWN']] = 1
            print(atom, "went to unknown column")

    return one_hot_matrix

def min_max_normalization(matrix):
    """
    Perform Min-Max normalization on a given matrix.

    Parameters:
    matrix (np.ndarray): The input matrix to be normalized.

    Returns:
    np.ndarray: The normalized matrix with values scaled to the range [0, 1].
    """
    # Compute the minimum and maximum values for the matrix
    min_val = np.min(matrix)
    max_val = np.max(matrix)

    # Apply Min-Max normalization formula
    normalized_matrix = (matrix - min_val) / (max_val - min_val)

    return normalized_matrix

def create_graphs(files, dataset_name):
    for file in files:
        protein_pdb_df = PandasPdb().read_pdb(file)
        protein_pdb_df.df.keys()
        protein = protein_pdb_df.df['ATOM']
        protein = protein[~protein['atom_name'].str.startswith('H')] # don't use hydrogen

        protein_coords = protein[['x_coord', 'y_coord', 'z_coord']].values
        protein_centroid = protein_coords.mean(axis=0)

        max_atoms = 150
        output_dir = f"{dataset_name}/{dataset_name}-graphs-5A"
        os.makedirs(output_dir, exist_ok=True)

        files = sorted(files, key=natural_sort_key)

        ligand_df = PandasPdb().read_pdb(file)
        ligand_df.df.keys()
        ligand = ligand_df.df['HETATM']
        ligand = ligand[ligand['residue_name']=="CLR"]
        x = list(set(zip(ligand['residue_number'], ligand['chain_id'])))

        #get the most inward residue
        min_distance = float('inf')
        closest_clr = None

        all_ligands = []

        for residue_number, chain_id in x:
            clr_atoms = ligand[(ligand['residue_number'] == residue_number) & (ligand['chain_id'] == chain_id)]
            if clr_atoms.empty:
                continue

            clr_coords = clr_atoms[['x_coord', 'y_coord', 'z_coord']].values
            clr_centroid = clr_coords.mean(axis=0)
            
            distance = np.linalg.norm(protein_centroid - clr_centroid)
            
            if distance < min_distance:
                min_distance = distance
                closest_clr = (residue_number, chain_id)

            grid_list_ = grid_list(clr_atoms)

            all_ligands.append(filtering_proteins(protein, grid_list_))

        ligand_ = ligand[(ligand['residue_number'] == closest_clr[0]) & (ligand['chain_id'] == closest_clr[1])]
        grid_list_ = grid_list(ligand_)

        filtered_atoms = filtering_proteins(protein, grid_list_)

        if not filtered_atoms.empty:
            # Save to pdb
            filtered_pdb = PandasPdb()
            filtered_pdb.df['ATOM'] = filtered_atoms
            base_name = os.path.basename(file)

            filtered_pdb_path = f"{dataset_name}/filtered-{dataset_name}-pdbs/{base_name[:4]}-filtered.pdb"
            os.makedirs(os.path.dirname(filtered_pdb_path), exist_ok=True)
            filtered_pdb.to_pdb(path=filtered_pdb_path, records=None, gz=False, append_newline=True)

        pdb_df = pdb_to_dataframe(filtered_pdb_path)
        encoded_matrix = one_hot_encoding(pdb_df)
        inverse_distance = compute_inverse_pairwise_distances(pdb_df) # don't need to normalize since gat notebook already does that
        
        combined_matrix = inverse_distance @ encoded_matrix # for gnn
        combined_matrix = min_max_normalization(combined_matrix)

        num_atoms = inverse_distance.shape[0]

        if num_atoms > max_atoms:
            print(f"{file} has {num_atoms} atoms, exceeding the limit of {max_atoms}")
            #raise Exception("Too many atoms!")
            continue
        
        combined_matrix = np.pad(combined_matrix, ((0, max_atoms - num_atoms), (0, 0)), mode='constant') # padding for gnn

        # Save to file
        base_name = os.path.splitext(os.path.basename(file))[0]
        output_path = os.path.join(output_dir, f"{base_name}_graphs.npy")

        np.save(output_path, { # for gat and gcn
            'inverse_distance': inverse_distance,
            'encoded_matrix': encoded_matrix
        })

        output_file = f"{dataset_name}/{dataset_name}-graph-5A/{base_name}_combined_matrix.npy" # for gnn
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        np.save(output_file, combined_matrix)