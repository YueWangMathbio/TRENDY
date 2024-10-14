"""
the training file for TRENDY method, first half
use Ktstar and then use TE(k=1) model, trained on all time points
given Kt, this model can convert it into Kt'
then we use K0 and Kt' to calculate A_1
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
import time
from evaluation import auroc_auprc
import random
from previous_methods.wendy_solver import RegRelSolver
from models import CovRev
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")
best_perform = 0.0

head_num = 4
layer_num = 7

# Set device based on GPU availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

# Load input and target data, reshape and stack datasets
def load_data():
    x_data = []
    y_data = []
    data_ranges = list(range(0, 100))
    for i in data_ranges:
        ds_num = f'{data_ranges[i]:02d}'  # File name formatting
        x_file = os.path.join(data_folder, f'dataset_{ds_num}_total_cov.npy')
        y_file = os.path.join(data_folder, f'dataset_{ds_num}_total_revcov.npy')

        x = np.load(x_file)[:, 1:, :, :]  # Extract required parts
        y = np.load(y_file)
        x = np.transpose(x, (1, 0, 2, 3)).reshape(-1, x.shape[2], x.shape[3])
        y = np.transpose(y, (1, 0, 2, 3)).reshape(-1, y.shape[2], y.shape[3])
        
        x_data.append(x)
        y_data.append(y)
    
    x_data = np.vstack(x_data)
    y_data = np.vstack(y_data)
    
    return torch.tensor(x_data, dtype=torch.float32, device=device), torch.tensor(y_data, dtype=torch.float32, device=device)

x_train, y_train = load_data()

repeat_num = (10, 1, 1)
k0_val = np.load(os.path.join(data_folder, 'dataset_val_total_cov.npy'))[:, 0, :, :]  
k0_val = np.tile(k0_val, repeat_num)
kt_val = np.load(os.path.join(data_folder, 'dataset_val_total_cov.npy'))[:, 1:, :, :]
kt_val = np.transpose(kt_val, (1, 0, 2, 3)).reshape(-1, kt_val.shape[2], kt_val.shape[3])
A_val = np.load(os.path.join(data_folder, 'dataset_val_total_A.npy'))  
A_val = np.tile(A_val, repeat_num)
wendy_val = np.load(os.path.join(data_folder, 'dataset_val_total_wendy.npy'))
wendy_val = np.transpose(wendy_val, (1, 0, 2, 3)).reshape(-1, wendy_val.shape[2], wendy_val.shape[3])
kt_val = torch.tensor(kt_val, dtype=torch.float32, device=device) 

# Create data loaders for training
batch_size = 64
train_dataset = torch.utils.data.TensorDataset(x_train, y_train)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# Instantiate model, loss function, and optimizer
model = CovRev(n_gene=10, d_model=64, nhead=head_num, num_layers=layer_num, dropout=0.1).to(device)

# Function for GRN matrix calculation using RegRelSolver
def wendy_cov(k0, kt):
    gene_num = len(k0)
    lam = 0.0  
    weight = np.ones((gene_num, gene_num))
    np.fill_diagonal(weight, 0.0)
    solver = RegRelSolver(k0, kt, lam, weight)
    my_at = solver.fit()
    return np.around(my_at, decimals=3)

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop with validation, early stopping, and timing
num_epochs = 100
model.train()

best_val_score = -float('inf')
patience = 10
epochs_without_improvement = 0

val_num = A_val.shape[0]
initial_val_auroc_sum = 0.0
initial_val_auprc_sum = 0.0
for j in range(val_num):
    auroc, auprc = auroc_auprc(A_val[j], wendy_val[j])
    initial_val_auroc_sum += auroc
    initial_val_auprc_sum += auprc

initial_val_auroc = initial_val_auroc_sum / val_num
initial_val_auprc = initial_val_auprc_sum / val_num
print(f'Initial (before training) Val AUROC: {initial_val_auroc:.4f}, Val AUPRC: {initial_val_auprc:.4f}')

# Training with validation and early stopping
for epoch in range(num_epochs):
    start_time = time.time()
    running_loss = 0.0
    
    with tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}", unit="batch") as progress_bar:
        for inputs, targets in progress_bar:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
    
    # Validation step
    model.eval()
    with torch.no_grad():
        val_auroc_sum = 0.0
        val_auprc_sum = 0.0
        kt_star = model(kt_val).cpu().numpy()
        
        for i in range(val_num):
            y_pred = wendy_cov(k0_val[i], kt_star[i])
            auroc, auprc = auroc_auprc(A_val[i], y_pred)
            val_auroc_sum += auroc
            val_auprc_sum += auprc
        
        val_auroc = val_auroc_sum / val_num
        val_auprc = val_auprc_sum / val_num
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
        
torch.save(model.state_dict(), "trendy_1.pth")

# in the following, use the above model to calculate Kt' and then calculate A_1

model.eval()
set_seed(42)  

# Define batch processing function for the wendy_cov calculation
def wendy_cov_batch(k0, kt):
    group_num, gene_num, _ = k0.shape
    res = np.zeros((group_num, gene_num, gene_num))
    lam = 0.0  # Regularization coefficient
    weight = np.ones((gene_num, gene_num))
    np.fill_diagonal(weight, 0.0)  # Ignore diagonal elements for matching
    
    for i in range(group_num):
        solver = RegRelSolver(k0[i], kt[i], lam, weight)  # Calculate A using the solver
        res[i] = np.around(solver.fit(), decimals=3)  # Save the calculated GRN matrix
    return res

data_folder = 'total_data_10'
rev_wendy_folder = 'rev_wendy_all_10'

# Process training data across all datasets
data_ranges = list(range(0, 100))
for i in data_ranges:
    ds_num = f'{data_ranges[i]:02d}'  # Format for file names
    xp_file = os.path.join(data_folder, f'dataset_{ds_num}_total_cov.npy')
    
    k0 = np.load(xp_file)[:, 0, :, :]  # Initial time point data
    kt = np.load(xp_file)[:, 1:, :, :]  # Time point data starting from the second
    kt = np.transpose(kt, (1, 0, 2, 3))  # Transpose to iterate over each time point
    
    kt_star_list = []
    rev_wendy_list = []
    
    # Process each time slice for the dataset
    for j in range(10):
        kt_curr = torch.tensor(kt[j], dtype=torch.float32, device=device)
        kt_star = model(kt_curr).cpu().detach().numpy()  # Model output for kt_star
        rev_wendy = wendy_cov_batch(k0, kt_star)  # Calculate rev_wendy
        kt_star_list.append(kt_star)
        rev_wendy_list.append(rev_wendy)
    
    # Rearrange and save output files
    kt_star_list = np.transpose(kt_star_list, (1, 0, 2, 3))
    rev_wendy_list = np.transpose(rev_wendy_list, (1, 0, 2, 3))
    np.save(os.path.join(rev_wendy_folder, f'dataset_{ds_num}_total_revwendy.npy'), rev_wendy_list) # the A_1 matrix

# Process validation data
k0 = np.load(os.path.join(data_folder, 'dataset_val_total_cov.npy'))[:, 0, :, :]
kt = np.load(os.path.join(data_folder, 'dataset_val_total_cov.npy'))[:, 1:, :, :]
kt = np.transpose(kt, (1, 0, 2, 3))

kt_star_list = []
rev_wendy_list = []

for j in range(10):
    kt_curr = torch.tensor(kt[j], dtype=torch.float32, device=device)
    kt_star = model(kt_curr).cpu().detach().numpy()
    rev_wendy = wendy_cov_batch(k0, kt_star)
    kt_star_list.append(kt_star)
    rev_wendy_list.append(rev_wendy)

kt_star_list = np.transpose(kt_star_list, (1, 0, 2, 3))
rev_wendy_list = np.transpose(rev_wendy_list, (1, 0, 2, 3))

np.save(os.path.join(rev_wendy_folder, 'dataset_val_total_revwendy.npy'), rev_wendy_list)

# Process test data
k0 = np.load(os.path.join(data_folder, 'dataset_test_total_cov.npy'))[:, 0, :, :]
kt = np.load(os.path.join(data_folder, 'dataset_test_total_cov.npy'))[:, 1:, :, :]
kt = np.transpose(kt, (1, 0, 2, 3))

kt_star_list = []
rev_wendy_list = []

for j in range(10):
    kt_curr = torch.tensor(kt[j], dtype=torch.float32, device=device)
    kt_star = model(kt_curr).cpu().detach().numpy()
    rev_wendy = wendy_cov_batch(k0, kt_star)
    kt_star_list.append(kt_star)
    rev_wendy_list.append(rev_wendy)

kt_star_list = np.transpose(kt_star_list, (1, 0, 2, 3))
rev_wendy_list = np.transpose(rev_wendy_list, (1, 0, 2, 3))

np.save(os.path.join(rev_wendy_folder, 'dataset_test_total_revwendy.npy'), rev_wendy_list)

