import glob
from FilterAtomsGraphs import create_graphs
import torch
import torch.nn as nn
import numpy as np
from torch_geometric.data import Data
from torch_geometric.nn import GATConv, global_mean_pool
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool
from EvaluationMethods import apply_labeling_from_spies, get_spy_info, show_evaluation_results
from collections import Counter
import os

test_files_path = "/home/alexhernandez/transmembranebindingAI/Data/UnsplitData/ivanfiles" # paste path of directory of files you want to evaluate
test_files = glob.glob(f"{test_files_path}/*.pdb")

output_dir = "CholBindOutput" # name the output dir of results

# put corresponding model paths here
gat_models_path = "/home/alexhernandez/transmembranebindingAI/Models/Cholesterol/GAT"
gnn_models_path = "/home/alexhernandez/transmembranebindingAI/Models/Cholesterol/GNN"
gcn_models_path = "/home/alexhernandez/transmembranebindingAI/Models/Cholesterol/GCN"

print("Preprocessing files")
create_graphs(test_files, output_dir)
print("Finished preprocessing")

gat_and_gcn_evaluation_path = f"{output_dir}/{output_dir}-graphs-5A" # gat and gcn use same input files
gnn_evaluation_path = f"{output_dir}/{output_dir}-graph-5A"

print("Performing GAT Analysis")

# GAT model for batched data
class GAT(torch.nn.Module):
    def __init__(self, in_channels, out_channels, dropout_p=0.1):
        super().__init__()
        self.gat = GATConv(in_channels, out_channels, heads=1, concat=True, edge_dim=1)
        self.pool = global_mean_pool  # Can also use global_max_pool or global_add_pool
        self.dropout = nn.Dropout(p=dropout_p)
        self.norm = nn.BatchNorm1d(out_channels)
        self.linear = torch.nn.Linear(out_channels, 1)

    def forward(self, x, edge_index, edge_attr, batch):
        out, attn_weights = self.gat(x, edge_index, edge_attr, return_attention_weights=True)
        out = self.dropout(out)
        out = self.pool(out, batch)  # Pool over nodes in each graph
        out = self.norm(out)
        out = self.dropout(out) 
        out = self.linear(out)
        return out, attn_weights

def organize_graph_and_add_weight(file_path, label):
    data = np.load(file_path, allow_pickle=True).item()
    inverse_distance = data['inverse_distance']
    encoded_matrix = data['encoded_matrix']

    x = torch.tensor(encoded_matrix, dtype=torch.float32)
    adj = torch.tensor(inverse_distance, dtype=torch.float32)

    # Normalize adjacency (row-normalize)
    adj = adj / (adj.sum(dim=1, keepdim=True) + 1e-8)

    # Create edge_index and edge weights
    edge_index = (adj > 0).nonzero(as_tuple=False).t()
    edge_weight = adj[adj > 0]

    y = torch.tensor([label], dtype=torch.float32)
    
    return Data(x=x, edge_index=edge_index, edge_attr=edge_weight, y=y)
    
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

models_exp1 = []
models_exp2 = []
models_exp3 = []
models_exp4 = []
models_exp5 = []

k = 50

for i in range (1, (k + 1)):
    model_exp1 = GAT(in_channels=37, out_channels=32).to(device)

    model_exp2 = GAT(in_channels=37, out_channels=32).to(device)

    model_exp3 = GAT(in_channels=37, out_channels=32).to(device)

    model_exp4 = GAT(in_channels=37, out_channels=32).to(device)

    model_exp5 = GAT(in_channels=37, out_channels=32).to(device)

    model_path = f"{gat_models_path}/GATModels-5A_exp1v2/Models/model_bin_{i}.pth" 
    model_exp1.load_state_dict(torch.load(model_path, map_location=device))

    model_path = f"{gat_models_path}/GATModels-5A_exp2v2/Models/model_bin_{i}.pth" 
    model_exp2.load_state_dict(torch.load(model_path, map_location=device))

    model_path = f"{gat_models_path}/GATModels-5A_exp3v2/Models/model_bin_{i}.pth" 
    model_exp3.load_state_dict(torch.load(model_path, map_location=device))

    model_path = f"{gat_models_path}/GATModels-5A_exp4v2/Models/model_bin_{i}.pth" 
    model_exp4.load_state_dict(torch.load(model_path, map_location=device))

    model_path = f"{gat_models_path}/GATModels-5A_exp5v2/Models/model_bin_{i}.pth" 
    model_exp5.load_state_dict(torch.load(model_path, map_location=device))

    model_exp1.eval()
    model_exp2.eval()
    model_exp3.eval()
    model_exp4.eval()
    model_exp5.eval()

    models_exp1.append(model_exp1)
    models_exp2.append(model_exp2)
    models_exp3.append(model_exp3)
    models_exp4.append(model_exp4)
    models_exp5.append(model_exp5)

def get_df(dir, models):
    rows = []
    files = sorted(glob.glob(f"{dir}/*.npy"))

    for file in files:
        model_probs = []

        # Process file for graph input
        graph = organize_graph_and_add_weight(file, label=0).to(device)
        non_padded_rows = graph.x.size(0)

        for model in models:
            with torch.no_grad():
                out, _ = model(
                    graph.x,
                    graph.edge_index,
                    graph.edge_attr,
                    batch=torch.zeros(graph.x.size(0), dtype=torch.long).to(device)
                )
                prob = torch.sigmoid(out).item()
                model_probs.append(prob)

        # Mean & std across models
        prediction_mean = float(np.mean(model_probs))
        prediction_std = float(np.std(model_probs))

        rows.append({
            "filename": file,
            "average_score": prediction_mean,
            "score_std": prediction_std,
            "number_atoms": int(non_padded_rows),
        })

    return pd.DataFrame(rows)

csv_output_exp1 = f"{gat_models_path}/GATModels-5A_exp1v2/NewResults/SpyCaptureRates.csv"
csv_output_exp2 = f"{gat_models_path}/GATModels-5A_exp2v2/NewResults/SpyCaptureRates.csv"
csv_output_exp3 = f"{gat_models_path}/GATModels-5A_exp3v2/NewResults/SpyCaptureRates.csv"
csv_output_exp4 = f"{gat_models_path}/GATModels-5A_exp4v2/NewResults/SpyCaptureRates.csv"
csv_output_exp5 = f"{gat_models_path}/GATModels-5A_exp5v2/NewResults/SpyCaptureRates.csv"

import pandas as pd

mean_score_exp1, percentile_50_exp1, percentile_25_exp1, percentile_75_exp1, min_score_exp1, max_score_exp1 = get_spy_info(csv_output_exp1)
mean_score_exp2, percentile_50_exp2, percentile_25_exp2, percentile_75_exp2, min_score_exp2, max_score_exp2 = get_spy_info(csv_output_exp2)
mean_score_exp3, percentile_50_exp3, percentile_25_exp3, percentile_75_exp3, min_score_exp3, max_score_exp3 = get_spy_info(csv_output_exp3)
mean_score_exp4, percentile_50_exp4, percentile_25_exp4, percentile_75_exp4, min_score_exp4, max_score_exp4 = get_spy_info(csv_output_exp4)
mean_score_exp5, percentile_50_exp5, percentile_25_exp5, percentile_75_exp5, min_score_exp5, max_score_exp5 = get_spy_info(csv_output_exp5)

gat_df1 = get_df(gat_and_gcn_evaluation_path, models_exp1)
gat_df2 = get_df(gat_and_gcn_evaluation_path, models_exp2)
gat_df3 = get_df(gat_and_gcn_evaluation_path, models_exp3)
gat_df4 = get_df(gat_and_gcn_evaluation_path, models_exp4)
gat_df5 = get_df(gat_and_gcn_evaluation_path, models_exp5)

gat_df1, min_max_results_exp1, percentile_results_exp1, threshold_results_exp1 = apply_labeling_from_spies(gat_df1, mean_score_exp1, percentile_50_exp1, percentile_25_exp1, percentile_75_exp1, min_score_exp1, max_score_exp1)
gat_df2, min_max_results_exp2, percentile_results_exp2, threshold_results_exp2 = apply_labeling_from_spies(gat_df2, mean_score_exp2, percentile_50_exp2, percentile_25_exp2, percentile_75_exp2, min_score_exp2, max_score_exp2)
gat_df3, min_max_results_exp3, percentile_results_exp3, threshold_results_exp3 = apply_labeling_from_spies(gat_df3, mean_score_exp3, percentile_50_exp3, percentile_25_exp3, percentile_75_exp3, min_score_exp3, max_score_exp3)
gat_df4, min_max_results_exp4, percentile_results_exp4, threshold_results_exp4 = apply_labeling_from_spies(gat_df4, mean_score_exp4, percentile_50_exp4, percentile_25_exp4, percentile_75_exp4, min_score_exp4, max_score_exp4)
gat_df5, min_max_results_exp5, percentile_results_exp5, threshold_results_exp5 = apply_labeling_from_spies(gat_df5, mean_score_exp5, percentile_50_exp5, percentile_25_exp5, percentile_75_exp5, min_score_exp5, max_score_exp5)

min_max_all = np.array([
    min_max_results_exp1,
    min_max_results_exp2,
    min_max_results_exp3,
    min_max_results_exp4,
    min_max_results_exp5
])

percentile_all = np.array([
    percentile_results_exp1,
    percentile_results_exp2,
    percentile_results_exp3,
    percentile_results_exp4,
    percentile_results_exp5
])

threshold_all = np.array([
    threshold_results_exp1,
    threshold_results_exp2,
    threshold_results_exp3,
    threshold_results_exp4,
    threshold_results_exp5
])

#show_evaluation_results(min_max_all, percentile_all, threshold_all)
print("GAT Analysis is finished")

print("Performing GCN Analysis")

class GCN(nn.Module):
    def __init__(self, input_dim):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(input_dim, 32)
        self.bn1 = nn.BatchNorm1d(32)
        
        self.conv2 = GCNConv(32, 64)
        self.bn2 = nn.BatchNorm1d(64)
        
        self.conv3 = GCNConv(64, 128)
        self.bn3 = nn.BatchNorm1d(128)

        self.dropout_gcn = nn.Dropout(0.2)
        self.dropout = nn.Dropout(0.6)
        
        self.fc1 = nn.Linear(128, 64)
        self.out = nn.Linear(64, 1)

    def forward(self, data):
        x, edge_index, edge_weight, batch = data.x, data.edge_index, data.edge_attr, data.batch

        x = self.conv1(x, edge_index, edge_weight)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.dropout_gcn(x)

        x = self.conv2(x, edge_index, edge_weight)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.dropout_gcn(x)

        x = self.conv3(x, edge_index, edge_weight)
        x = self.bn3(x)
        x = F.relu(x)
        x = self.dropout_gcn(x)

        # Global pooling to get graph-level representation
        x = global_mean_pool(x, batch)

        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.out(x)

        return x
    
def organize_graph_and_add_weight(file_path, label):
    data = np.load(file_path, allow_pickle=True).item()
    inverse_distance = data['inverse_distance']
    encoded_matrix = data['encoded_matrix']

    x = torch.tensor(encoded_matrix, dtype=torch.float32)
    adj = torch.tensor(inverse_distance, dtype=torch.float32)

    # Normalize adjacency (row-normalize)
    #adj = adj / (adj.sum(dim=1, keepdim=True) + 1e-8)

    # Create edge_index and edge weights
    edge_index = (adj > 0).nonzero(as_tuple=False).t()
    edge_weight = adj[adj > 0]

    y = torch.tensor([label], dtype=torch.float32)
    
    return Data(x=x, edge_index=edge_index, edge_attr=edge_weight, y=y)
    
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

models_exp1 = []
models_exp2 = []
models_exp3 = []
models_exp4 = []
models_exp5 = []

k = 50

for i in range (1, (k + 1)):
    model_exp1 = GCN(input_dim=37).to(device)

    model_exp2 = GCN(input_dim=37).to(device)

    model_exp3 = GCN(input_dim=37).to(device)

    model_exp4 = GCN(input_dim=37).to(device)

    model_exp5 = GCN(input_dim=37).to(device)

    model_path = f"{gcn_models_path}/GCN-5A_Exp1/Models/model_bin_{i}.pth" 
    model_exp1.load_state_dict(torch.load(model_path, map_location=device))

    model_path = f"{gcn_models_path}/GCN-5A_Exp2/Models/model_bin_{i}.pth" 
    model_exp2.load_state_dict(torch.load(model_path, map_location=device))

    model_path = f"{gcn_models_path}/GCN-5A_Exp3/Models/model_bin_{i}.pth" 
    model_exp3.load_state_dict(torch.load(model_path, map_location=device))

    model_path = f"{gcn_models_path}/GCN-5A_Exp4/Models/model_bin_{i}.pth" 
    model_exp4.load_state_dict(torch.load(model_path, map_location=device))

    model_path = f"{gcn_models_path}/GCN-5A_Exp5/Models/model_bin_{i}.pth" 
    model_exp5.load_state_dict(torch.load(model_path, map_location=device))

    model_exp1.eval()
    model_exp2.eval()
    model_exp3.eval()
    model_exp4.eval()
    model_exp5.eval()

    models_exp1.append(model_exp1)
    models_exp2.append(model_exp2)
    models_exp3.append(model_exp3)
    models_exp4.append(model_exp4)
    models_exp5.append(model_exp5)

def get_df(dir, models):
    rows = []
    files = sorted(glob.glob(f"{dir}/*.npy"))

    for file in files:
        model_probs = []

        # Build graph
        graph = organize_graph_and_add_weight(file, label=0).to(device)
        graph.batch = torch.zeros(graph.num_nodes, dtype=torch.long).to(device)
        non_padded_rows = graph.x.size(0)

        for model in models:
            with torch.no_grad():
                out = model(graph)
                prob = torch.sigmoid(out).item()
                model_probs.append(prob)

        # Mean & std across models
        prediction_mean = float(np.mean(model_probs))
        prediction_std = float(np.std(model_probs))

        rows.append({
            "filename": file,
            "average_score": prediction_mean,
            "score_std": prediction_std,
            "number_atoms": int(non_padded_rows),
        })

    return pd.DataFrame(rows)

csv_output_exp1 = f"{gcn_models_path}/GCN-5A_Exp1/Results/SpyCaptureRates.csv"
csv_output_exp2 = f"{gcn_models_path}/GCN-5A_Exp2/Results/SpyCaptureRates.csv"
csv_output_exp3 = f"{gcn_models_path}/GCN-5A_Exp3/Results/SpyCaptureRates.csv"
csv_output_exp4 = f"{gcn_models_path}/GCN-5A_Exp4/Results/SpyCaptureRates.csv"
csv_output_exp5 = f"{gcn_models_path}/GCN-5A_Exp5/Results/SpyCaptureRates.csv"

mean_score_exp1, percentile_50_exp1, percentile_25_exp1, percentile_75_exp1, min_score_exp1, max_score_exp1 = get_spy_info(csv_output_exp1)
mean_score_exp2, percentile_50_exp2, percentile_25_exp2, percentile_75_exp2, min_score_exp2, max_score_exp2 = get_spy_info(csv_output_exp2)
mean_score_exp3, percentile_50_exp3, percentile_25_exp3, percentile_75_exp3, min_score_exp3, max_score_exp3 = get_spy_info(csv_output_exp3)
mean_score_exp4, percentile_50_exp4, percentile_25_exp4, percentile_75_exp4, min_score_exp4, max_score_exp4 = get_spy_info(csv_output_exp4)
mean_score_exp5, percentile_50_exp5, percentile_25_exp5, percentile_75_exp5, min_score_exp5, max_score_exp5 = get_spy_info(csv_output_exp5)

gcn_df1 = get_df(gat_and_gcn_evaluation_path, models_exp1)
gcn_df2 = get_df(gat_and_gcn_evaluation_path, models_exp2)
gcn_df3 = get_df(gat_and_gcn_evaluation_path, models_exp3)
gcn_df4 = get_df(gat_and_gcn_evaluation_path, models_exp4)
gcn_df5 = get_df(gat_and_gcn_evaluation_path, models_exp5)

gcn_df1, min_max_results_exp1, percentile_results_exp1, threshold_results_exp1 = apply_labeling_from_spies(gcn_df1, mean_score_exp1, percentile_50_exp1, percentile_25_exp1, percentile_75_exp1, min_score_exp1, max_score_exp1)
gcn_df2, min_max_results_exp2, percentile_results_exp2, threshold_results_exp2 = apply_labeling_from_spies(gcn_df2, mean_score_exp2, percentile_50_exp2, percentile_25_exp2, percentile_75_exp2, min_score_exp2, max_score_exp2)
gcn_df3, min_max_results_exp3, percentile_results_exp3, threshold_results_exp3 = apply_labeling_from_spies(gcn_df3, mean_score_exp3, percentile_50_exp3, percentile_25_exp3, percentile_75_exp3, min_score_exp3, max_score_exp3)
gcn_df4, min_max_results_exp4, percentile_results_exp4, threshold_results_exp4 = apply_labeling_from_spies(gcn_df4, mean_score_exp4, percentile_50_exp4, percentile_25_exp4, percentile_75_exp4, min_score_exp4, max_score_exp4)
gcn_df5, min_max_results_exp5, percentile_results_exp5, threshold_results_exp5 = apply_labeling_from_spies(gcn_df5, mean_score_exp5, percentile_50_exp5, percentile_25_exp5, percentile_75_exp5, min_score_exp5, max_score_exp5)

min_max_all = np.array([
    min_max_results_exp1,
    min_max_results_exp2,
    min_max_results_exp3,
    min_max_results_exp4,
    min_max_results_exp5
])

percentile_all = np.array([
    percentile_results_exp1,
    percentile_results_exp2,
    percentile_results_exp3,
    percentile_results_exp4,
    percentile_results_exp5
])

threshold_all = np.array([
    threshold_results_exp1,
    threshold_results_exp2,
    threshold_results_exp3,
    threshold_results_exp4,
    threshold_results_exp5
])

#show_evaluation_results(min_max_all, percentile_all, threshold_all)

print("GCN Analysis is finished")

print("Performing GNN Analysis")

# Define the 2D CNN model in PyTorch
class CNN2D(nn.Module):
    def __init__(self, input_channels):
        super(CNN2D, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU()
        )
        self.pool1 = nn.MaxPool2d(2, 2)
        
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        self.pool2 = nn.MaxPool2d(2, 2)
        
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU()
        )
        self.pool3 = nn.MaxPool2d(2, 2)
        
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(128 * 4 * 18, 128)  # Adjust based on input size
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(128, 1)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.pool1(x)
        
        x = self.conv2(x)
        x = self.pool2(x)
        
        x = self.conv3(x)
        x = self.pool3(x)
        
        x = self.flatten(x)
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)

        return x
    
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

models_exp1 = []
models_exp2 = []
models_exp3 = []
models_exp4 = []
models_exp5 = []

k = 50

for i in range (1, (k + 1)):
    model_exp1 = CNN2D(input_channels=1).to(device)
    model_exp1 = nn.DataParallel(model_exp1)

    model_exp2 = CNN2D(input_channels=1).to(device)
    model_exp2 = nn.DataParallel(model_exp2)

    model_exp3 = CNN2D(input_channels=1).to(device)
    model_exp3 = nn.DataParallel(model_exp3)

    model_exp4 = CNN2D(input_channels=1).to(device)
    model_exp4 = nn.DataParallel(model_exp4)

    model_exp5 = CNN2D(input_channels=1).to(device)
    model_exp5 = nn.DataParallel(model_exp5)

    model_path = f"{gnn_models_path}/GNN-5A_Exp1/Models/model_bin_{i}.pth" 
    model_exp1.load_state_dict(torch.load(model_path, map_location=device))

    model_path = f"{gnn_models_path}/GNN-5A_Exp2/Models/model_bin_{i}.pth" 
    model_exp2.load_state_dict(torch.load(model_path, map_location=device))

    model_path = f"{gnn_models_path}/GNN-5A_Exp3/Models/model_bin_{i}.pth" 
    model_exp3.load_state_dict(torch.load(model_path, map_location=device))

    model_path = f"{gnn_models_path}/GNN-5A_Exp4/Models/model_bin_{i}.pth" 
    model_exp4.load_state_dict(torch.load(model_path, map_location=device))

    model_path = f"{gnn_models_path}/GNN-5A_Exp5/Models/model_bin_{i}.pth" 
    model_exp5.load_state_dict(torch.load(model_path, map_location=device))

    model_exp1.eval()
    model_exp2.eval()
    model_exp3.eval()
    model_exp4.eval()
    model_exp5.eval()

    models_exp1.append(model_exp1)
    models_exp2.append(model_exp2)
    models_exp3.append(model_exp3)
    models_exp4.append(model_exp4)
    models_exp5.append(model_exp5)

def evaluate_file(model, file_path, threshold=0.5):
    grid = np.load(file_path)

    if grid.ndim == 2:
        non_padded_rows = np.sum(np.any(grid != 0, axis=(1)))
    else:
        raise ValueError(f"Unexpected grid shape: {grid.shape}")
    
    grid_tensor = torch.tensor(grid, dtype=torch.float32).unsqueeze(0).unsqueeze(0)  # Add batch and channel dims
    grid_tensor = grid_tensor.to(device)
    
    model.eval()
    with torch.no_grad():
        output = model(grid_tensor).squeeze(1)  

    prob = torch.sigmoid(output).item()

    predicted_class = int(prob >= threshold)

    return predicted_class, prob, non_padded_rows

def get_df(dir, models):
    """
    Evaluate all models on .npy files using evaluate_file(model, file).
    Returns a DataFrame with columns:
    filename, average_score, score_std, number_atoms
    """

    rows = []
    files = sorted(glob.glob(f"{dir}/*.npy"))

    for file in files:
        model_probs = []
        non_padded_rows = None

        for model in models:
            predicted_class, prob, n_atoms = evaluate_file(model, file)
            model_probs.append(prob)
            non_padded_rows = n_atoms   # same for all models

        # Mean and std of model probabilities
        prediction_mean = float(np.mean(model_probs))
        prediction_std = float(np.std(model_probs))

        rows.append({
            "filename": file,
            "average_score": prediction_mean,
            "score_std": prediction_std,
            "number_atoms": int(non_padded_rows),
        })

    return pd.DataFrame(rows)

csv_output_exp1 = f"{gnn_models_path}/GNN-5A_Exp1/NewResults/SpyCaptureRates.csv"
csv_output_exp2 = f"{gnn_models_path}/GNN-5A_Exp2/NewResults/SpyCaptureRates.csv"
csv_output_exp3 = f"{gnn_models_path}/GNN-5A_Exp3/NewResults/SpyCaptureRates.csv"
csv_output_exp4 = f"{gnn_models_path}/GNN-5A_Exp4/NewResults/SpyCaptureRates.csv"
csv_output_exp5 = f"{gnn_models_path}/GNN-5A_Exp5/NewResults/SpyCaptureRates.csv"

import pandas as pd

mean_score_exp1, percentile_50_exp1, percentile_25_exp1, percentile_75_exp1, min_score_exp1, max_score_exp1 = get_spy_info(csv_output_exp1)
mean_score_exp2, percentile_50_exp2, percentile_25_exp2, percentile_75_exp2, min_score_exp2, max_score_exp2 = get_spy_info(csv_output_exp2)
mean_score_exp3, percentile_50_exp3, percentile_25_exp3, percentile_75_exp3, min_score_exp3, max_score_exp3 = get_spy_info(csv_output_exp3)
mean_score_exp4, percentile_50_exp4, percentile_25_exp4, percentile_75_exp4, min_score_exp4, max_score_exp4 = get_spy_info(csv_output_exp4)
mean_score_exp5, percentile_50_exp5, percentile_25_exp5, percentile_75_exp5, min_score_exp5, max_score_exp5 = get_spy_info(csv_output_exp5)

gnn_df1 = get_df(gnn_evaluation_path, models_exp1)
gnn_df2 = get_df(gnn_evaluation_path, models_exp2)
gnn_df3 = get_df(gnn_evaluation_path, models_exp3)
gnn_df4 = get_df(gnn_evaluation_path, models_exp4)
gnn_df5 = get_df(gnn_evaluation_path, models_exp5)

gnn_df1, min_max_results_exp1, percentile_results_exp1, threshold_results_exp1 = apply_labeling_from_spies(gnn_df1, mean_score_exp1, percentile_50_exp1, percentile_25_exp1, percentile_75_exp1, min_score_exp1, max_score_exp1)
gnn_df2, min_max_results_exp2, percentile_results_exp2, threshold_results_exp2 = apply_labeling_from_spies(gnn_df2, mean_score_exp2, percentile_50_exp2, percentile_25_exp2, percentile_75_exp2, min_score_exp2, max_score_exp2)
gnn_df3, min_max_results_exp3, percentile_results_exp3, threshold_results_exp3 = apply_labeling_from_spies(gnn_df3, mean_score_exp3, percentile_50_exp3, percentile_25_exp3, percentile_75_exp3, min_score_exp3, max_score_exp3)
gnn_df4, min_max_results_exp4, percentile_results_exp4, threshold_results_exp4 = apply_labeling_from_spies(gnn_df4, mean_score_exp4, percentile_50_exp4, percentile_25_exp4, percentile_75_exp4, min_score_exp4, max_score_exp4)
gnn_df5, min_max_results_exp5, percentile_results_exp5, threshold_results_exp5 = apply_labeling_from_spies(gnn_df5, mean_score_exp5, percentile_50_exp5, percentile_25_exp5, percentile_75_exp5, min_score_exp5, max_score_exp5)

min_max_all = np.array([
    min_max_results_exp1,
    min_max_results_exp2,
    min_max_results_exp3,
    min_max_results_exp4,
    min_max_results_exp5
])

percentile_all = np.array([
    percentile_results_exp1,
    percentile_results_exp2,
    percentile_results_exp3,
    percentile_results_exp4,
    percentile_results_exp5
])

threshold_all = np.array([
    threshold_results_exp1,
    threshold_results_exp2,
    threshold_results_exp3,
    threshold_results_exp4,
    threshold_results_exp5
])

print("GNN Analysis is finished")
#show_evaluation_results(min_max_all, percentile_all, threshold_all)

# finally able to summarize the results
results_dfs = {
    "GAT": [gat_df1, gat_df2, gat_df3, gat_df4, gat_df5],
    "GCN": [gcn_df1, gcn_df2, gcn_df3, gcn_df4, gcn_df5],
    "GNN": [gnn_df1, gnn_df2, gnn_df3, gnn_df4, gnn_df5],
}


def majority_label(labels):
    """Return the most common percentile label."""
    return Counter(labels).most_common(1)[0][0]

def build_full_summary_from_dfs(results_dict):
    frames = []

    # ---------- Attach model + experiment and concat ----------
    for model_name, df_list in results_dict.items():
        for exp_idx, df in enumerate(df_list, start=1):
            df_copy = df.copy()
            df_copy["model"] = model_name
            df_copy["experiment"] = exp_idx
            frames.append(df_copy)

    all_df = pd.concat(frames, ignore_index=True)

    # ---------- Add protein_id from first 4 chars of basename ----------
    all_df["protein_id"] = all_df["filename"].apply(
        lambda x: os.path.basename(str(x))[:4]
    )

    # ---------- Summary counts over percentile_label ----------
    percentile_counts = (
        all_df.groupby(["model", "experiment", "percentile_label"])
        .size()
        .reset_index(name="count")
    )

    # ---------- Score summary per model/experiment ----------
    score_summary = (
        all_df.groupby(["model", "experiment"])
        .agg(
            n_samples=("filename", "count"),
            avg_score_mean=("average_score", "mean"),
            avg_score_std=("average_score", "std"),
        )
        .reset_index()
    )

    # ---------- Per-protein summary across experiments & filenames ----------
    summary_rows = []

    for protein_id, sub_df in all_df.groupby("protein_id"):
        row = {"protein_id": protein_id}

        for model in ["GAT", "GCN", "GNN"]:
            model_df = sub_df[sub_df["model"] == model]

            if len(model_df) > 0:
                # Average score across all experiments & files for this protein and model
                row[f"{model}_avg_score"] = model_df["average_score"].mean()

                # Majority-vote percentile label across those rows
                row[f"{model}_percentile_label"] = majority_label(
                    model_df["percentile_label"].tolist()
                )
            else:
                row[f"{model}_avg_score"] = None
                row[f"{model}_percentile_label"] = None

        summary_rows.append(row)

    summary_df = pd.DataFrame(summary_rows)

    return all_df, percentile_counts, score_summary, summary_df

# -------------------- RUN --------------------
all_df, percentile_counts, score_summary, summary_df = build_full_summary_from_dfs(results_dfs)

all_df.to_csv(f"{output_dir}/summary.csv", index=False)
print("Summary of results saved to csv")

print("Percentile label counts:")
print(percentile_counts)

print("\nScore summary:")
print(score_summary)

print("\nPer-file summary (first few rows):")
print(summary_df.head())