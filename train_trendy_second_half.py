"""
the training file for TRENDY method, second half
use A_1, K0, Kt and then use TE(k=3) model, trained on all time points
we can obtain A_2, the final output of TRENDY method
"""
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
import time
import random
from evaluation import auroc_auprc
from models import TripleGRN
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")

# Set the device to GPU if available
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

set_seed(42)  # Initialize seed

data_folder = 'total_data_10'
rev_wendy_folder = 'rev_wendy_all_10'

# Load input and target data, reshape, and stack datasets for all time points
def load_data():
    x_data = []
    x_k0_data = []
    x_ktstar_data = []
    y_data = []
    data_ranges = list(range(0, 100))
    
    for time_point in range(10):
        for i in data_ranges:
            ds_num = f'{data_ranges[i]:02d}'  # File name formatting
            x_file = os.path.join(rev_wendy_folder, f'dataset_{ds_num}_total_revwendy.npy')
            y_file = os.path.join(data_folder, f'dataset_{ds_num}_total_A.npy')
            xp_file = os.path.join(data_folder, f'dataset_{ds_num}_total_cov.npy')
            
            x = np.load(x_file)[:, time_point, :, :]  # Load revwendy data for each time point
            x_k0 = np.load(xp_file)[:, 0, :, :]  # Initial state data
            x_ktstar = np.load(xp_file)[:, time_point+1, :, :]  # Load subsequent time point data
            y = np.load(y_file)  # Load target data (GRN matrix)
            
            x_data.append(x)
            x_k0_data.append(x_k0)
            x_ktstar_data.append(x_ktstar)
            y_data.append(y)
    
    x_data = np.vstack(x_data)
    x_k0_data = np.vstack(x_k0_data)
    x_ktstar_data = np.vstack(x_ktstar_data)
    y_data = np.vstack(y_data)
    
    return (torch.tensor(x_data, dtype=torch.float32, device=device), 
            torch.tensor(x_k0_data, dtype=torch.float32, device=device), 
            torch.tensor(x_ktstar_data, dtype=torch.float32, device=device), 
            torch.tensor(y_data, dtype=torch.float32, device=device))

x_train, x_k0_train, x_ktstar_train, y_train = load_data()

# Load and stack validation data for all time points
x_val, x_k0_val, x_ktstar_val, y_val = [], [], [], []
for time_point in range(10):
    x_val.append(np.load(os.path.join(rev_wendy_folder, 'dataset_val_total_revwendy.npy'))[:, time_point, :, :])
    x_k0_val.append(np.load(os.path.join(data_folder, 'dataset_val_total_cov.npy'))[:, 0, :, :])
    x_ktstar_val.append(np.load(os.path.join(data_folder, 'dataset_val_total_cov.npy'))[:, time_point+1, :, :])
    y_val.append(np.load(os.path.join(data_folder, 'dataset_val_total_A.npy')))

x_val = np.vstack(x_val)
x_k0_val = np.vstack(x_k0_val)
x_ktstar_val = np.vstack(x_ktstar_val)
y_val = np.vstack(y_val)

x_val = torch.tensor(x_val, dtype=torch.float32, device=device)
x_k0_val = torch.tensor(x_k0_val, dtype=torch.float32, device=device)
x_ktstar_val = torch.tensor(x_ktstar_val, dtype=torch.float32, device=device)
y_val = torch.tensor(y_val, dtype=torch.float32, device=device)  

# Create data loader for training
batch_size = 64
train_dataset = torch.utils.data.TensorDataset(x_train, x_k0_train, x_ktstar_train, y_train)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# Instantiate model, loss function, and optimizer
model = TripleGRN(n_gene=10, d_model=64, nhead=head_num, num_layers=layer_num, dropout=0.1).to(device)

# Custom loss function excluding diagonal elements
def mse_without_diagonal(y_pred, y_true):
    mask = torch.ones_like(y_true, dtype=torch.bool)
    mask[:, torch.arange(y_pred.shape[1]), torch.arange(y_pred.shape[2])] = False
    y_true_no_diag = y_true[mask].view(y_true.shape[0], -1)
    y_pred_no_diag = y_pred[mask].view(y_pred.shape[0], -1)
    return nn.functional.mse_loss(y_pred_no_diag, y_true_no_diag)

criterion = mse_without_diagonal
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop with validation, early stopping, and timing
num_epochs = 100
model.train()

best_val_score = -float('inf')
patience = 10
epochs_without_improvement = 0

# Initial validation AUROC and AUPRC on raw data
initial_val_auroc_sum = 0.0
initial_val_auprc_sum = 0.0
for i in range(x_val.size(0)):
    y_true = y_val[i].cpu().numpy()
    x_input = x_val[i].cpu().numpy()
    auroc, auprc = auroc_auprc(y_true, x_input)
    initial_val_auroc_sum += auroc
    initial_val_auprc_sum += auprc

initial_val_auroc = initial_val_auroc_sum / x_val.size(0)
initial_val_auprc = initial_val_auprc_sum / x_val.size(0)
print(f'Initial (before training) Val AUROC: {initial_val_auroc:.4f}, Val AUPRC: {initial_val_auprc:.4f}')

# Training with validation and early stopping
for epoch in range(num_epochs):
    start_time = time.time()
    running_loss = 0.0
    
    with tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}", unit="batch") as progress_bar:
        for inputs0, inputs1, inputs2, targets in progress_bar:
            optimizer.zero_grad()
            outputs = model(inputs0, inputs1, inputs2)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
    
    # Validation step
    model.eval()
    with torch.no_grad():
        val_auroc_sum = 0.0
        val_auprc_sum = 0.0
        y_pred_all = model(x_val, x_k0_val, x_ktstar_val).cpu().numpy()
        
        for i in range(x_val.size(0)):
            y_true = y_val[i].cpu().numpy()
            y_pred = y_pred_all[i]
            auroc, auprc = auroc_auprc(y_true, y_pred)
            val_auroc_sum += auroc
            val_auprc_sum += auprc
        
        val_auroc = val_auroc_sum / x_val.size(0)
        val_auprc = val_auprc_sum / x_val.size(0)
        val_score = val_auroc + val_auprc

        # Early stopping check
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

torch.save(model.state_dict(), "trendy_2.pth")
