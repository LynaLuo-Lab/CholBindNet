# 3D CNN Model
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import Dataset, DataLoader
import glob
import matplotlib.pyplot as plt
from collections import Counter
import os
import torch.nn.functional as F
import pandas as pd
from TrackAtoms import match_atoms_to_pdb
from Bio.PDB import PDBParser
import re

# 3D CNN Model
class CNN3D(nn.Module):
    def __init__(self):
        super(CNN3D, self).__init__()
        
        self.conv0 = nn.Conv3d(in_channels=37, out_channels=64, kernel_size=1, stride=1, padding=0) # play around with output channels
        self.conv1 = nn.Conv3d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool3d(kernel_size=2, stride=2, padding=0)
        self.conv2 = nn.Conv3d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.pool2 = nn.MaxPool3d(kernel_size=2, stride=2, padding=0)

        #self.dropout_conv = nn.Dropout3d(p=0.05)
        
        # After two pooling layers, spatial dimensions reduce from 40x40x40 -> 5x5x5
        self.fc1 = nn.Linear(128 * 3 * 3 * 3, 256)  # Try increasing over 256
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 2)  # Assuming 1 output for docking status/position

        #self.dropout_fc = nn.Dropout(p=0.15)
        
    def forward(self, x):
        # Forward pass through Conv layers
        x = self.pool(torch.relu(self.conv0(x)))  # Conv0 -> ReLU -> Pooling
        #x = self.dropout_conv(x)
        x = self.pool(torch.relu(self.conv1(x)))  # Conv1 -> ReLU -> Pooling
        x = self.pool2(torch.relu(self.conv2(x)))  # Conv2 -> ReLU -> Pooling

        # Flatten the input for fully connected layers
        x = x.view(-1, 128 * 3 * 3 * 3)
        
        # Forward pass through fully connected layers
        x = torch.relu(self.fc1(x)) #use tanh activation
        #x = self.dropout_fc(x)
        x = torch.relu(self.fc2(x))
        x = torch.nn.functional.softmax(self.fc3(x), dim=1)  # Final layer (output layer)
        #x = torch.clamp(x, min=1e-7, max=1 - 1e-7)  # Clamp outputs to avoid extreme values
        
        return x

def get_atom_info_from_pdb(pdb_file):
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("mol", pdb_file)
    atoms = []

    for model in structure:
        for chain in model:
            for residue in chain:
                for atom in residue:
                    if atom.is_disordered():
                        for alt_atom in atom.disordered_get_list():
                            atoms.append(alt_atom)
                    else:
                        atoms.append(atom)

    return atoms

def natural_sort_key(s):
    """Function to sort strings in a natural alphanumeric order."""
    return [int(text) if text.isdigit() else text.lower() for text in re.split(r'(\d+)', s)]

# === Grad-CAM for 3D CNN ===
class GradCAM3D:
    def __init__(self, model, target_layer):
        self.model = model
        self.model.eval()
        self.target_layer = target_layer
        
        self.gradients = None
        self.activations = None

        self.hook_handles = []
        self._register_hooks()

    def _register_hooks(self):
        def forward_hook(module, input, output):
            self.activations = output.detach()

        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()

        self.hook_handles.append(self.target_layer.register_forward_hook(forward_hook))
        self.hook_handles.append(self.target_layer.register_full_backward_hook(backward_hook))  # updated

    def generate_cam(self, input_tensor, target_class=None):
        output = self.model(input_tensor)

        if target_class is None:
            target_class = torch.argmax(output, dim=1).item()

        self.model.zero_grad()
        one_hot = torch.zeros_like(output)
        one_hot[0][target_class] = 1
        output.backward(gradient=one_hot)

        grads = self.gradients[0]  # (C, D, H, W)
        activations = self.activations[0]  # (C, D, H, W)

        weights = torch.mean(grads.view(grads.size(0), -1), dim=1)
        cam = torch.zeros(activations.shape[1:], dtype=torch.float32)
        for i, w in enumerate(weights):
            cam += w * activations[i]

        cam = F.relu(cam)
        cam -= cam.min()
        cam /= (cam.max() + 1e-8)

        return cam.cpu().numpy()

    def remove_hooks(self):
        for handle in self.hook_handles:
            handle.remove()

subtype_counts_per_exp = []
residue_counts_per_exp = []

def save_counters_to_csv(counter_list, out_csv):
    """
    Save a list[Counter] (one Counter per experiment) to a single CSV where
    rows = experiments and columns = labels (missing labels filled with 0).
    """
    # Union of all labels across experiments
    all_labels = sorted(set().union(*[set(c.keys()) for c in counter_list])) if counter_list else []
    rows = []
    for i, c in enumerate(counter_list, start=1):
        row = {lbl: int(c.get(lbl, 0)) for lbl in all_labels}
        row["__experiment__"] = i
        rows.append(row)
    df = pd.DataFrame(rows).set_index("__experiment__")
    df.to_csv(out_csv, index=True)

def load_counters_from_csv(csv_path):
    """
    Load the CSV saved by save_counters_to_csv back into a list[Counter],
    one Counter per experiment (row). Works with plot_mean_std() unchanged.
    """
    df = pd.read_csv(csv_path, index_col="__experiment__")
    # Ensure ints (in case CSV round-trips as floats)
    df = df.fillna(0)
    if not df.empty:
        df = df.astype(int)
    counters = []
    for _, row in df.iterrows():
        # Only keep nonzero entries to keep Counters clean
        c = Counter({col: int(val) for col, val in row.items() if int(val) != 0})
        counters.append(c)
    return counters
    
# preload models
all_models = []

models_exp1 = []
model_paths = sorted(glob.glob("../../../Models/Cholesterol/3DCNN/3DCholesterolModels-st_exp1/Models/*.pth"), key=natural_sort_key)
for mp in model_paths:
    m = CNN3D()
    m.load_state_dict(torch.load(mp, weights_only=True))
    # Create GradCAM with CPU model (hooks bind by object, device can change later)
    gc = GradCAM3D(m, m.conv2)  # adjust layer if needed
    models_exp1.append((m, gc))
all_models.append(models_exp1)

models_exp2 = []
model_paths = sorted(glob.glob("../../../Models/Cholesterol/3DCNN/3DCholesterolModels-st_exp2/Models/*.pth"), key=natural_sort_key)
for mp in model_paths:
    m = CNN3D()
    m.load_state_dict(torch.load(mp, weights_only=True))
    # Create GradCAM with CPU model (hooks bind by object, device can change later)
    gc = GradCAM3D(m, m.conv2)  # adjust layer if needed
    models_exp2.append((m, gc))
all_models.append(models_exp2)

models_exp3 = []
model_paths = sorted(glob.glob("../../../Models/Cholesterol/3DCNN/3DCholesterolModels-st_exp3/Models/*.pth"), key=natural_sort_key)
for mp in model_paths:
    m = CNN3D()
    m.load_state_dict(torch.load(mp, weights_only=True))
    # Create GradCAM with CPU model (hooks bind by object, device can change later)
    gc = GradCAM3D(m, m.conv2)  # adjust layer if needed
    models_exp3.append((m, gc))
all_models.append(models_exp3)

models_exp4 = []
model_paths = sorted(glob.glob("../../../Models/Cholesterol/3DCNN/3DCholesterolModels-st_exp4/Models/*.pth"), key=natural_sort_key)
for mp in model_paths:
    m = CNN3D()
    m.load_state_dict(torch.load(mp, weights_only=True))
    # Create GradCAM with CPU model (hooks bind by object, device can change later)
    gc = GradCAM3D(m, m.conv2)  # adjust layer if needed
    models_exp4.append((m, gc))
all_models.append(models_exp4)

models_exp5 = []
model_paths = sorted(glob.glob("../../../Models/Cholesterol/3DCNN/3DCholesterolModels-st_exp5/Models/*.pth"), key=natural_sort_key)
for mp in model_paths:
    m = CNN3D()
    m.load_state_dict(torch.load(mp, weights_only=True))
    # Create GradCAM with CPU model (hooks bind by object, device can change later)
    gc = GradCAM3D(m, m.conv2)  # adjust layer if needed
    models_exp5.append((m, gc))
all_models.append(models_exp5)

def plot_mean_std(counter_list, title, xlabel):
    # union of all labels
    labels = sorted(set().union(*[set(c.keys()) for c in counter_list]))
    # build matrix E x L (experiments x labels)
    M = np.array([[c.get(lbl, 0) for lbl in labels] for c in counter_list], dtype=float)
    means = M.mean(axis=0)
    stds  = M.std(axis=0, ddof=1) if M.shape[0] > 1 else np.zeros_like(means)

    # sort indices by mean (descending)
    sorted_idx = np.argsort(means)[::-1]
    labels = [labels[i] for i in sorted_idx]
    means = means[sorted_idx]
    stds  = stds[sorted_idx]

    # plot
    x = np.arange(len(labels))
    plt.rcParams.update({"font.size": 18})  # increase global font size
    plt.figure(figsize=(12,5))              # smaller horizontal size
    plt.bar(x, means, yerr=stds, capsize=4)
    plt.xticks(x, labels, rotation=45, ha='right')
    plt.title(title, fontsize=20)
    plt.xlabel(xlabel, fontsize=18)
    plt.ylabel("Count (mean ± std)", fontsize=18)
    plt.tight_layout()
    plt.show()

def save_feature_scores(all_feature_scores, mean_tensor, std_tensor,
                        per_exp_csv="all_feature_scores_per_exp.csv",
                        summary_csv="all_feature_scores_summary.csv",
                        feature_labels=None):
    """
    Save per-experiment feature scores and the final mean/std summary to CSVs.
    - all_feature_scores: list of torch tensors (per experiment, shape=(23,))
    - mean_tensor: torch tensor shape (23,)
    - std_tensor:  torch tensor shape (23,)
    """
    # Per-experiment matrix
    per_exp_data = torch.stack(all_feature_scores, dim=0).cpu().numpy()
    df_exp = pd.DataFrame(per_exp_data, columns=feature_labels)
    df_exp.index.name = "experiment"
    df_exp.to_csv(per_exp_csv)

    # Summary (mean ± std)
    df_summary = pd.DataFrame({
        "feature": feature_labels,
        "mean": mean_tensor.cpu().numpy(),
        "std": std_tensor.cpu().numpy()
    })
    df_summary.to_csv(summary_csv, index=False)
    print(f"Saved {per_exp_csv} and {summary_csv}")


def load_feature_scores(per_exp_csv="all_feature_scores_per_exp.csv",
                        summary_csv="all_feature_scores_summary.csv"):
    """
    Reload per-experiment feature scores and summary.
    Returns (all_feature_scores, mean_tensor, std_tensor, feature_labels).
    """
    # Per-experiment
    df_exp = pd.read_csv(per_exp_csv, index_col="experiment")
    feature_labels = df_exp.columns.tolist()
    all_feature_scores = [torch.tensor(row.values, dtype=torch.float32)
                          for _, row in df_exp.iterrows()]

    # Summary
    df_summary = pd.read_csv(summary_csv)
    mean_tensor = torch.tensor(df_summary["mean"].values, dtype=torch.float32)
    std_tensor  = torch.tensor(df_summary["std"].values, dtype=torch.float32)

    return all_feature_scores, mean_tensor, std_tensor, feature_labels

# === Load Input ===

#input_files = glob.glob("../../../Data/SplitData/Cholesterol/IvanTestSet/ivan-grid-st/positive/*.npy")
all_feature_scores = []

for exp_index, experiment_models in enumerate(all_models):
    input_files = glob.glob(f"../../../Data/SplitData/Cholesterol/cholesterol-grid-st_exp{exp_index + 1}/Test/Positive/*.npy")
    experiment_feature_scores = []
    for index, file_path in enumerate(input_files):
        if index % 10 == 0:
            print(f"{index}/{len(input_files)}")
        
        input_tensor = np.load(file_path)  # (30, 30, 30, 23)
        input_tensor = torch.from_numpy(input_tensor).float().permute(3, 0, 1, 2).unsqueeze(0)  # (1, 23, 30, 30, 30)
        
        for model, gc in experiment_models:
            device = next(model.parameters()).device
            x = input_tensor.to(device)

            # --- Grad-CAM heatmap for this input & model ---
            # enable grads because Grad-CAM does a backward pass internally
            with torch.enable_grad():
                cam_np = gc.generate_cam(x, target_class=None)  # shape: (D_cam, H_cam, W_cam), values ~ [0,1]
            cam = torch.from_numpy(cam_np).to(device).float()

            # --- Resize CAM to match input spatial size if needed ---
            D, H, W = x.shape[2], x.shape[3], x.shape[4]
            if cam.shape != (D, H, W):
                cam = F.interpolate(
                    cam.unsqueeze(0).unsqueeze(0),  # (1,1,Dc,Hc,Wc)
                    size=(D, H, W),
                    mode="trilinear",
                    align_corners=False
                ).squeeze(0).squeeze(0)

            # Safety re-normalization (keeps CAM in [0,1], avoids degenerate cases)
            cam_min = cam.min()
            cam_max = cam.max()
            if (cam_max - cam_min) > 1e-8:
                cam = (cam - cam_min) / (cam_max - cam_min)
            else:
                cam = torch.zeros_like(cam)

            # --- Per-feature importance using CAM as a spatial weight ---
            # x: (1, C, D, H, W), cam: (D, H, W)
            # broadcast cam over channels and batch
            weighted = (x[0] * cam.unsqueeze(0))  # (C, D, H, W)
            feature_scores = weighted.abs().mean(dim=(1, 2, 3))  # (C,)

            experiment_feature_scores.append(feature_scores.detach().cpu())

    # === Average across 50 models ===
    print(len(experiment_feature_scores[0]))
    feature_scores_stack = torch.stack(experiment_feature_scores)  #
    feature_scores_mean = feature_scores_stack.sum(dim=0)  # 
    all_feature_scores.append(feature_scores_mean)

# === Mean & Std across the 5 experiments ===
F = torch.stack(all_feature_scores, dim=0)            # (E,23)
all_feature_scores_mean = F.mean(dim=0)               # (23,)
all_feature_scores_std  = F.std(dim=0, unbiased=True) # (23,)

# === Optional: Top-k important features ===
topk = torch.topk(feature_scores_mean, k=5)
print("\nTop 5 Important Features:")
for i, (idx, val) in enumerate(zip(topk.indices.tolist(), topk.values.tolist()), 1):
    print(f"{i}. Feature {idx} → Importance Score")

#feature_scores_mean

feature_labels = [
    'C', 'CA', 'CB', 'CD', 'CD1', 'CD2', 'CE', 'CE1', 'CE2', 'CE3',
    'CG', 'CG1', 'CG2', 'CH2', 'CZ', 'CZ2', 'CZ3',

    # Oxygen (O) subtypes
    'O', 'OH', 'OD1', 'OD2', 'OE1', 'OE2', 'OG', 'OG1',

    # Nitrogen (N) subtypes
    'N', 'NE', 'NE1', 'NE2', 'ND1', 'ND2', 'NZ', 'NH1', 'NH2',

    # Sulfur (S) subtypes
    'SD', 'SG', 'OXT'
]

save_feature_scores(
    all_feature_scores,
    all_feature_scores_mean,
    all_feature_scores_std,
    per_exp_csv="all_feature_scores_per_exp_internal_cnn_gradcam_st.csv",
    summary_csv="all_feature_scores_summary_internal_cnn_gradcam_st.csv",
    feature_labels=feature_labels
)