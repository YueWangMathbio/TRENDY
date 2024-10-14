"""
the training file for NonlinearODEs-rev method
use nlode and then use TE(k=1) model, trained on all time points
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
import time
import random
from evaluation import auroc_auprc
from models import CovRev
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")

# Set device to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

head_num = 8
layer_num = 7

# Seed setting function for reproducibility
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)  # Seed initialization

data_folder = 'total_data_10'

# Function to load and stack input data
def load_data():
    x_data = []
    y_data = []
    data_ranges = list(range(0, 100))
    for i in data_ranges:
        ds_num = f'{data_ranges[i]:02d}'  # Format to match file naming convention
        x_file = os.path.join(data_folder, f'dataset_{ds_num}_total_nlode.npy')
        y_file = os.path.join(data_folder, f'dataset_{ds_num}_total_A.npy')
        x = np.load(x_file)
        y = np.load(y_file)  # Load the target matrices

        x_data.append(x)
        y_data.append(y)
    
    x_data = np.vstack(x_data)
    y_data = np.vstack(y_data)
    
    return torch.tensor(x_data, dtype=torch.float32, device=device), torch.tensor(y_data, dtype=torch.float32, device=device)

x_train, y_train = load_data()

# Load validation data
x_val = np.load(os.path.join(data_folder, 'dataset_val_total_nlode.npy'))
y_val = np.load(os.path.join(data_folder, 'dataset_val_total_A.npy'))
x_val = torch.tensor(x_val, dtype=torch.float32, device=device)
y_val = torch.tensor(y_val, dtype=torch.float32, device=device)  

# Create data loaders for training
batch_size = 64
train_dataset = torch.utils.data.TensorDataset(x_train, y_train)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# Model instantiation with specific parameters
model = CovRev(n_gene=10, d_model=64, nhead=head_num, num_layers=layer_num, dropout=0.1).to(device)

# Define custom MSE loss function excluding diagonal elements
def mse_without_diagonal(y_pred, y_true):
    mask = torch.ones_like(y_true, dtype=torch.bool)
    mask[:, torch.arange(y_pred.shape[1]), torch.arange(y_pred.shape[2])] = False

    y_true_no_diag = y_true[mask].view(y_true.shape[0], -1)
    y_pred_no_diag = y_pred[mask].view(y_pred.shape[0], -1)

    return nn.functional.mse_loss(y_pred_no_diag, y_true_no_diag)

criterion = mse_without_diagonal
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training parameters and early stopping setup
num_epochs = 100
model.train()

best_val_score = -float('inf')
patience = 10
epochs_without_improvement = 0

# Initial AUROC and AUPRC calculation on validation data
initial_val_auroc_sum = 0.0
initial_val_auprc_sum = 0.0
for i in range(x_val.size(0)):  # Iterate over each validation sample
    y_true = y_val[i].cpu().numpy()
    x_input = x_val[i].cpu().numpy()
    auroc, auprc = auroc_auprc(y_true, x_input)
    initial_val_auroc_sum += auroc
    initial_val_auprc_sum += auprc

initial_val_auroc = initial_val_auroc_sum / x_val.size(0)
initial_val_auprc = initial_val_auprc_sum / x_val.size(0)
print(f'Initial (before training) Val AUROC: {initial_val_auroc:.4f}, Val AUPRC: {initial_val_auprc:.4f}')

# Training loop
for epoch in range(num_epochs):
    start_time = time.time()
    running_loss = 0.0
    
    with tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}", unit="batch") as progress_bar:
        for inputs, targets in progress_bar:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
    
    # Validation calculations
    model.eval()
    with torch.no_grad():
        val_auroc_sum = 0.0
        val_auprc_sum = 0.0
        y_pred_all = model(x_val).cpu().numpy()
        for i in range(x_val.size(0)):
            y_true = y_val[i].cpu().numpy()
            y_pred = y_pred_all[i]
            auroc, auprc = auroc_auprc(y_true, y_pred)
            val_auroc_sum += auroc
            val_auprc_sum += auprc
        
        val_auroc = val_auroc_sum / x_val.size(0)
        val_auprc = val_auprc_sum / x_val.size(0)
        val_score = val_auroc + val_auprc

        # Check for early stopping
        if val_score > best_val_score:
            best_val_score = val_score
            best_model_state = model.state_dict()
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

    epoch_time = time.time() - start_time
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}, Val AUROC: {val_auroc:.4f}, Val AUPRC: {val_auprc:.4f}, Time: {epoch_time:.2f}s')
    
    if epochs_without_improvement >= patience:
        print(f"Early stopping triggered after {epoch+1} epochs.")
        break

    model.train()

if best_model_state is not None:
    model.load_state_dict(best_model_state)
        
torch.save(model.state_dict(), "weights/nlode_rev.pth")
