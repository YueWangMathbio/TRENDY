"""
measure the performance of all eight methods on SINC data
"""

import torch
import numpy as np
import os
import random
from evaluation import auroc_auprc
from models import TripleGRN, DoubleGRN, CovRev
import warnings
warnings.filterwarnings("ignore")
# Set the device to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
        
set_seed(42)

# WENDY and TRENDY
data_folder = 'total_data_10'
rev_wendy_folder = 'rev_wendy_all_10'
x_ori = []
x_test = []
x_k0_test = []
x_ktstar_test = []
y_test = []
for time_point in range(10):
    x_ori.append(np.load(os.path.join(data_folder, 'dataset_test_total_wendy.npy'))[:, time_point, :, :])
    x_test.append(np.load(os.path.join(rev_wendy_folder, 'dataset_test_total_revwendy.npy'))[:, time_point, :, :])  
    x_k0_test.append(np.load(os.path.join(data_folder, 'dataset_test_total_cov.npy'))[:, 0, :, :])
    x_ktstar_test.append(np.load(os.path.join(data_folder, 'dataset_test_total_cov.npy'))[:, time_point+1, :, :])
    y_test.append(np.load(os.path.join(data_folder, 'dataset_test_total_A.npy')))  
x_ori = np.vstack(x_ori)
x_test = np.vstack(x_test)
x_k0_test = np.vstack(x_k0_test)
x_ktstar_test = np.vstack(x_ktstar_test)
y_test = np.vstack(y_test)
x_ori = torch.tensor(x_ori, dtype=torch.float32, device=device)
x_test = torch.tensor(x_test, dtype=torch.float32, device=device)
x_k0_test = torch.tensor(x_k0_test, dtype=torch.float32, device=device)
x_ktstar_test = torch.tensor(x_ktstar_test, dtype=torch.float32, device=device)
y_test = torch.tensor(y_test, dtype=torch.float32, device=device)  

model = TripleGRN(n_gene=10, d_model=64, nhead=8, num_layers=7, dropout=0.1).to(device)
model.load_state_dict(torch.load('weights/trendy_2.pth', map_location=device))
model.eval()

test_size = x_test.size(0) // 10
initial_auroc_list = []
initial_auprc_list = []
test_auroc_list = []
test_auprc_list = []
with torch.no_grad():
    for j in range(10):
        initial_auroc_sum = 0.0
        initial_auprc_sum = 0.0
        for i in range(test_size*j, test_size*(j+1)): 
            y_true = y_test[i].cpu().numpy() 
            x_input = x_ori[i].cpu().numpy()  
            auroc, auprc = auroc_auprc(y_true, x_input)
            initial_auroc_sum += auroc
            initial_auprc_sum += auprc
        
        initial_auroc = initial_auroc_sum / test_size
        initial_auprc = initial_auprc_sum / test_size
        initial_auroc_list.append(initial_auroc)
        initial_auprc_list.append(initial_auprc)
    
        # Model evaluation
        test_auroc_sum = 0.0
        test_auprc_sum = 0.0
        for i in range(test_size*j, test_size*(j+1)):  
            y_true = y_test[i].cpu().numpy()  
            y_pred = model(x_test[i].unsqueeze(0), x_k0_test[i].unsqueeze(0), x_ktstar_test[i].unsqueeze(0)).squeeze(0).cpu().numpy()  # Predict and squeeze to (10, 10)
            auroc, auprc = auroc_auprc(y_true, y_pred)
            test_auroc_sum += auroc
            test_auprc_sum += auprc
        
        test_auroc = test_auroc_sum / test_size
        test_auprc = test_auprc_sum / test_size
        test_auroc_list.append(test_auroc)
        test_auprc_list.append(test_auprc)
mean_initial_auroc = np.mean(initial_auroc_list)
mean_initial_auprc = np.mean(initial_auprc_list)
mean_test_auroc = np.mean(test_auroc_list)
mean_test_auprc = np.mean(test_auprc_list)
print('Initial AUROC: ')
print([float(f'{temp:.4f}') for temp in initial_auroc_list])
print('Initial AUPRC: ')
print([float(f'{temp:.4f}') for temp in initial_auprc_list])
print('Test AUROC: ')
print([float(f'{temp:.4f}') for temp in test_auroc_list])
print('Test AUPRC: ')
print([float(f'{temp:.4f}') for temp in test_auprc_list])
print(f'WENDY, AUROC, AUPRC: {mean_initial_auroc:.4f}, {mean_initial_auprc:.4f}')
print(f'TRENDY, AUROC, AUPRC: {mean_test_auroc:.4f}, {mean_test_auprc:.4f}')
"""
Initial AUROC: 
[0.7819, 0.7025, 0.6658, 0.6567, 0.6517, 0.6493, 0.6434, 0.6378, 0.6356, 0.6298]
Initial AUPRC: 
[0.7014, 0.6433, 0.6191, 0.6086, 0.6016, 0.596, 0.5904, 0.586, 0.5829, 0.5801]
Test AUROC: 
[0.9254, 0.89, 0.867, 0.862, 0.8527, 0.8422, 0.8354, 0.8305, 0.8246, 0.8177]
Test AUPRC: 
[0.8803, 0.826, 0.7879, 0.769, 0.7449, 0.7234, 0.7086, 0.699, 0.693, 0.687]
"""
# WENDY, AUROC, AUPRC: 0.6654, 0.6109
# TRENDY, AUROC, AUPRC: 0.8547, 0.7519


# GENIE3 and GENIE3-rev
set_seed(42)
data_folder = 'total_data_10'
x_test = []
x_k_test = []
y_test = []
for time_point in range(10):
    x_test.append(np.load(os.path.join(data_folder, 'dataset_test_total_genie.npy'))[:, time_point, :, :])  
    x_k_test.append(np.load(os.path.join(data_folder, 'dataset_test_total_cov.npy'))[:, time_point+1, :, :])
    y_test.append(np.load(os.path.join(data_folder, 'dataset_test_total_A.npy')))  
x_test = np.vstack(x_test)
x_k_test = np.vstack(x_k_test)
y_test = np.vstack(y_test)
x_test = torch.tensor(x_test, dtype=torch.float32, device=device)
x_k_test = torch.tensor(x_k_test, dtype=torch.float32, device=device)
y_test = torch.tensor(y_test, dtype=torch.float32, device=device)  

model = DoubleGRN(n_gene=10, d_model=64, nhead=8, num_layers=7, dropout=0.1).to(device)
model.load_state_dict(torch.load('weights/genie_rev.pth', map_location=device))
model.eval()

test_size = x_test.size(0) // 10
initial_auroc_list = []
initial_auprc_list = []
test_auroc_list = []
test_auprc_list = []
with torch.no_grad():
    for j in range(10):
        initial_auroc_sum = 0.0
        initial_auprc_sum = 0.0
        for i in range(test_size*j, test_size*(j+1)):  
            y_true = y_test[i].cpu().numpy()  
            x_input = x_test[i].cpu().numpy()  
            auroc, auprc = auroc_auprc(y_true, x_input)
            initial_auroc_sum += auroc
            initial_auprc_sum += auprc
        
        initial_auroc = initial_auroc_sum / test_size
        initial_auprc = initial_auprc_sum / test_size
        initial_auroc_list.append(initial_auroc)
        initial_auprc_list.append(initial_auprc)
    
        # Model evaluation
        test_auroc_sum = 0.0
        test_auprc_sum = 0.0
        for i in range(test_size*j, test_size*(j+1)):  
            y_true = y_test[i].cpu().numpy()  
            y_pred = model(x_test[i].unsqueeze(0), x_k_test[i].unsqueeze(0)).squeeze(0).cpu().numpy()  
            auroc, auprc = auroc_auprc(y_true, y_pred)
            test_auroc_sum += auroc
            test_auprc_sum += auprc
        
        test_auroc = test_auroc_sum / test_size
        test_auprc = test_auprc_sum / test_size        
        test_auroc_list.append(test_auroc)
        test_auprc_list.append(test_auprc)

mean_initial_auroc = np.mean(initial_auroc_list)
mean_initial_auprc = np.mean(initial_auprc_list)
mean_test_auroc = np.mean(test_auroc_list)
mean_test_auprc = np.mean(test_auprc_list)
print('Initial AUROC: ')
print([float(f'{temp:.4f}') for temp in initial_auroc_list])
print('Initial AUPRC: ')
print([float(f'{temp:.4f}') for temp in initial_auprc_list])
print('Test AUROC: ')
print([float(f'{temp:.4f}') for temp in test_auroc_list])
print('Test AUPRC: ')
print([float(f'{temp:.4f}') for temp in test_auprc_list])
print(f'GENIE3, AUROC, AUPRC: {mean_initial_auroc:.4f}, {mean_initial_auprc:.4f}')
print(f'GENIE3-rev, AUROC, AUPRC: {mean_test_auroc:.4f}, {mean_test_auprc:.4f}')
"""
Initial AUROC: 
[0.4384, 0.4374, 0.4427, 0.4403, 0.4264, 0.4079, 0.3858, 0.3704, 0.3588, 0.3536]
Initial AUPRC: 
[0.5538, 0.5387, 0.5281, 0.518, 0.5086, 0.4979, 0.4879, 0.48, 0.4757, 0.4728]
Test AUROC: 
[0.9382, 0.912, 0.888, 0.877, 0.8661, 0.8551, 0.8471, 0.8441, 0.8407, 0.8346]
Test AUPRC: 
[0.8984, 0.8512, 0.8107, 0.7853, 0.7606, 0.7368, 0.7188, 0.7094, 0.7027, 0.6982]
"""
# GENIE3, AUROC, AUPRC: 0.4062, 0.5061
# GENIE3-rev, AUROC, AUPRC: 0.8703, 0.7672


# SINCERITIES and SINCERITIES-rev
set_seed(42)
data_folder = 'total_data_10'
x_test = np.load(os.path.join(data_folder, 'dataset_test_total_sinc.npy'))
y_test = np.load(os.path.join(data_folder, 'dataset_test_total_A.npy'))
x_test = torch.tensor(x_test, dtype=torch.float32, device=device)
y_test = torch.tensor(y_test, dtype=torch.float32, device=device)  

model = CovRev(n_gene=10, d_model=64, nhead=8, num_layers=7, dropout=0.1).to(device)
model.load_state_dict(torch.load('weights/sinc_rev.pth', map_location=device))
model.eval()

test_size = x_test.size(0)
with torch.no_grad():
    initial_auroc_sum = 0.0
    initial_auprc_sum = 0.0
    for i in range(test_size):  
        y_true = y_test[i].cpu().numpy()  
        x_input = x_test[i].cpu().numpy() 
        auroc, auprc = auroc_auprc(y_true, x_input)
        initial_auroc_sum += auroc
        initial_auprc_sum += auprc
    
    initial_auroc = initial_auroc_sum / test_size
    initial_auprc = initial_auprc_sum / test_size
    print(f'SINCERITIES, AUROC, AUPRC: {initial_auroc:.4f}, {initial_auprc:.4f}')

    test_auroc_sum = 0.0
    test_auprc_sum = 0.0
    for i in range(test_size):  
        y_true = y_test[i].cpu().numpy()  
        y_pred = model(x_test[i].unsqueeze(0)).squeeze(0).cpu().numpy()  
        auroc, auprc = auroc_auprc(y_true, y_pred)
        test_auroc_sum += auroc
        test_auprc_sum += auprc
    
    test_auroc = test_auroc_sum / test_size
    test_auprc = test_auprc_sum / test_size
    print(f'SINCERITIES-rev, AUROC, AUPRC: {test_auroc:.4f}, {test_auprc:.4f}')
# SINCERITIES, AUROC, AUPRC: 0.6783, 0.5829
# SINCERITIES-rev, AUROC, AUPRC: 0.7964, 0.6637


# NonlinearODEs and NonlinearODEs-rev
set_seed(42)
data_folder = 'total_data_10'
x_test = np.load(os.path.join(data_folder, 'dataset_test_total_nlode.npy'))
y_test = np.load(os.path.join(data_folder, 'dataset_test_total_A.npy'))
x_test = torch.tensor(x_test, dtype=torch.float32, device=device)
y_test = torch.tensor(y_test, dtype=torch.float32, device=device)  

model = CovRev(n_gene=10, d_model=64, nhead=8, num_layers=7, dropout=0.1).to(device)
model.load_state_dict(torch.load('weights/nlode_rev.pth', map_location=device))
model.eval()

test_size = x_test.size(0)
with torch.no_grad():
    initial_auroc_sum = 0.0
    initial_auprc_sum = 0.0
    for i in range(test_size): 
        y_true = y_test[i].cpu().numpy()  
        x_input = x_test[i].cpu().numpy() 
        auroc, auprc = auroc_auprc(y_true, x_input)
        initial_auroc_sum += auroc
        initial_auprc_sum += auprc
    
    initial_auroc = initial_auroc_sum / test_size
    initial_auprc = initial_auprc_sum / test_size
    print(f'NonlinearODEs, AUROC, AUPRC: {initial_auroc:.4f}, {initial_auprc:.4f}')

    test_auroc_sum = 0.0
    test_auprc_sum = 0.0
    for i in range(test_size):  
        y_true = y_test[i].cpu().numpy()  
        y_pred = model(x_test[i].unsqueeze(0)).squeeze(0).cpu().numpy()  
        auroc, auprc = auroc_auprc(y_true, y_pred)
        test_auroc_sum += auroc
        test_auprc_sum += auprc
    
    test_auroc = test_auroc_sum / test_size
    test_auprc = test_auprc_sum / test_size
    print(f'NonlinearODEs-rev, AUROC, AUPRC: {test_auroc:.4f}, {test_auprc:.4f}')
# NonlinearODEs, AUROC, AUPRC: 0.5076, 0.5313
# NonlinearODEs-rev, AUROC, AUPRC: 0.5976, 0.5658





