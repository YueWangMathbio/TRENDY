"""
measure the performance of all 16 methods on SINC data
"""

import torch
import numpy as np
import os
import random
from evaluation import auroc_auprc
from models import TripleGRN, DoubleGRN, CovRev
from brane_alg import BRANE
from nd_alg import ND_regulatory
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
        

for counter in range(3):
    if counter == 0:
        namestr = '010'
    if counter == 1:
        namestr = '001'
    if counter == 2:
        namestr = '100'
    set_seed(42)
    print(namestr)
    
    # WENDY and TRENDY
    data_folder = 'SINC'
    rev_wendy_folder = 'SINC'
    x_ori = []
    x_test = []
    x_k0_test = []
    x_ktstar_test = []
    y_test = []
    for time_point in range(10):
        x_ori.append(np.load(os.path.join(data_folder, f'dataset_sigma{namestr}_total_wendy.npy'))[:, time_point, :, :])
        x_test.append(np.load(os.path.join(rev_wendy_folder, f'dataset_sigma{namestr}_total_revwendy.npy'))[:, time_point, :, :])  
        x_k0_test.append(np.load(os.path.join(data_folder, f'dataset_sigma{namestr}_total_cov.npy'))[:, 0, :, :])
        x_ktstar_test.append(np.load(os.path.join(data_folder, f'dataset_sigma{namestr}_total_cov.npy'))[:, time_point+1, :, :])
        y_test.append(np.load(os.path.join(data_folder, f'dataset_sigma{namestr}_total_A.npy')))  
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
    nd_auroc_list = []
    nd_auprc_list = []
    bc_auroc_list = []
    bc_auprc_list = []
    with torch.no_grad():
        for j in range(10):
            initial_auroc_sum = 0.0
            initial_auprc_sum = 0.0
            nd_auroc_sum = 0.0
            nd_auprc_sum = 0.0
            bc_auroc_sum = 0.0
            bc_auprc_sum = 0.0
            for i in range(test_size*j, test_size*(j+1)): 
                y_true = y_test[i].cpu().numpy() 
                x_input = x_ori[i].cpu().numpy()  
                auroc, auprc = auroc_auprc(y_true, x_input)
                initial_auroc_sum += auroc
                initial_auprc_sum += auprc
                A_0_nd = ND_regulatory(x_input)
                A_0_bc = BRANE(x_input)
                auroc, auprc = auroc_auprc(y_true, A_0_nd)
                nd_auroc_sum += auroc
                nd_auprc_sum += auprc
                auroc, auprc = auroc_auprc(y_true, A_0_bc)    
                bc_auroc_sum += auroc
                bc_auprc_sum += auprc
            
            initial_auroc = initial_auroc_sum / test_size
            initial_auprc = initial_auprc_sum / test_size
            initial_auroc_list.append(initial_auroc)
            initial_auprc_list.append(initial_auprc)
            nd_auroc = nd_auroc_sum / test_size
            nd_auprc = nd_auprc_sum / test_size
            nd_auroc_list.append(nd_auroc)
            nd_auprc_list.append(nd_auprc)
            bc_auroc = bc_auroc_sum / test_size
            bc_auprc = bc_auprc_sum / test_size
            bc_auroc_list.append(bc_auroc)
            bc_auprc_list.append(bc_auprc)
        
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
    mean_nd_auroc = np.mean(nd_auroc_list)
    mean_nd_auprc = np.mean(nd_auprc_list)
    mean_bc_auroc = np.mean(bc_auroc_list)
    mean_bc_auprc = np.mean(bc_auprc_list)
    print(f'{mean_initial_auroc:.4f}, {mean_initial_auprc:.4f}, {mean_test_auroc:.4f}, {mean_test_auprc:.4f}, {mean_nd_auroc:.4f}, {mean_nd_auprc:.4f}, {mean_bc_auroc:.4f}, {mean_bc_auprc:.4f}')
    
    
    # GENIE3 and GENIE3-rev
    set_seed(42)
    data_folder = 'SINC'
    x_test = []
    x_k_test = []
    y_test = []
    for time_point in range(10):
        x_test.append(np.load(os.path.join(data_folder, f'dataset_sigma{namestr}_total_genie.npy'))[:, time_point, :, :])  
        x_k_test.append(np.load(os.path.join(data_folder, f'dataset_sigma{namestr}_total_cov.npy'))[:, time_point+1, :, :])
        y_test.append(np.load(os.path.join(data_folder, f'dataset_sigma{namestr}_total_A.npy')))  
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
    nd_auroc_list = []
    nd_auprc_list = []
    bc_auroc_list = []
    bc_auprc_list = []
    with torch.no_grad():
        for j in range(10):
            initial_auroc_sum = 0.0
            initial_auprc_sum = 0.0
            nd_auroc_sum = 0.0
            nd_auprc_sum = 0.0
            bc_auroc_sum = 0.0
            bc_auprc_sum = 0.0
            for i in range(test_size*j, test_size*(j+1)):  
                y_true = y_test[i].cpu().numpy()  
                x_input = x_test[i].cpu().numpy()  
                auroc, auprc = auroc_auprc(y_true, x_input)
                initial_auroc_sum += auroc
                initial_auprc_sum += auprc
                A_0_nd = ND_regulatory(x_input)
                A_0_bc = BRANE(x_input)
                auroc, auprc = auroc_auprc(y_true, A_0_nd)
                nd_auroc_sum += auroc
                nd_auprc_sum += auprc
                auroc, auprc = auroc_auprc(y_true, A_0_bc)    
                bc_auroc_sum += auroc
                bc_auprc_sum += auprc
            
            initial_auroc = initial_auroc_sum / test_size
            initial_auprc = initial_auprc_sum / test_size
            initial_auroc_list.append(initial_auroc)
            initial_auprc_list.append(initial_auprc)
            nd_auroc = nd_auroc_sum / test_size
            nd_auprc = nd_auprc_sum / test_size
            nd_auroc_list.append(nd_auroc)
            nd_auprc_list.append(nd_auprc)
            bc_auroc = bc_auroc_sum / test_size
            bc_auprc = bc_auprc_sum / test_size
            bc_auroc_list.append(bc_auroc)
            bc_auprc_list.append(bc_auprc)
        
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
    mean_nd_auroc = np.mean(nd_auroc_list)
    mean_nd_auprc = np.mean(nd_auprc_list)
    mean_bc_auroc = np.mean(bc_auroc_list)
    mean_bc_auprc = np.mean(bc_auprc_list)
    print(f'{mean_initial_auroc:.4f}, {mean_initial_auprc:.4f}, {mean_test_auroc:.4f}, {mean_test_auprc:.4f}, {mean_nd_auroc:.4f}, {mean_nd_auprc:.4f}, {mean_bc_auroc:.4f}, {mean_bc_auprc:.4f}')
    
    
    
    # SINCERITIES and SINCERITIES-rev
    set_seed(42)
    data_folder = 'SINC'
    x_test = np.load(os.path.join(data_folder, f'dataset_sigma{namestr}_total_sinc.npy'))
    y_test = np.load(os.path.join(data_folder, f'dataset_sigma{namestr}_total_A.npy'))
    x_test = torch.tensor(x_test, dtype=torch.float32, device=device)
    y_test = torch.tensor(y_test, dtype=torch.float32, device=device)  
    
    model = CovRev(n_gene=10, d_model=64, nhead=8, num_layers=7, dropout=0.1).to(device)
    model.load_state_dict(torch.load('weights/sinc_rev.pth', map_location=device))
    model.eval()
    
    test_size = x_test.size(0)
    with torch.no_grad():
        initial_auroc_sum = 0.0
        initial_auprc_sum = 0.0
        nd_auroc_sum = 0.0
        nd_auprc_sum = 0.0
        bc_auroc_sum = 0.0
        bc_auprc_sum = 0.0
        for i in range(test_size):  
            y_true = y_test[i].cpu().numpy()  
            x_input = x_test[i].cpu().numpy() 
            auroc, auprc = auroc_auprc(y_true, x_input)
            initial_auroc_sum += auroc
            initial_auprc_sum += auprc
            A_0_nd = ND_regulatory(x_input)
            A_0_bc = BRANE(x_input)
            auroc, auprc = auroc_auprc(y_true, A_0_nd)
            nd_auroc_sum += auroc
            nd_auprc_sum += auprc
            auroc, auprc = auroc_auprc(y_true, A_0_bc)    
            bc_auroc_sum += auroc
            bc_auprc_sum += auprc
        
        initial_auroc = initial_auroc_sum / test_size
        initial_auprc = initial_auprc_sum / test_size
        nd_auroc = nd_auroc_sum / test_size
        nd_auprc = nd_auprc_sum / test_size
        bc_auroc = bc_auroc_sum / test_size
        bc_auprc = bc_auprc_sum / test_size
    
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
        print(f'{initial_auroc:.4f}, {initial_auprc:.4f}, {test_auroc:.4f}, {test_auprc:.4f}, {nd_auroc:.4f}, {nd_auprc:.4f}, {bc_auroc:.4f}, {bc_auprc:.4f}')
    
    # NonlinearODEs and NonlinearODEs-rev
    set_seed(42)
    data_folder = 'SINC'
    x_test = np.load(os.path.join(data_folder, f'dataset_sigma{namestr}_total_nlode.npy'))
    y_test = np.load(os.path.join(data_folder, f'dataset_sigma{namestr}_total_A.npy'))
    x_test = torch.tensor(x_test, dtype=torch.float32, device=device)
    y_test = torch.tensor(y_test, dtype=torch.float32, device=device)  
    
    model = CovRev(n_gene=10, d_model=64, nhead=8, num_layers=7, dropout=0.1).to(device)
    model.load_state_dict(torch.load('weights/nlode_rev.pth', map_location=device))
    model.eval()
    
    test_size = x_test.size(0)
    with torch.no_grad():
        initial_auroc_sum = 0.0
        initial_auprc_sum = 0.0
        nd_auroc_sum = 0.0
        nd_auprc_sum = 0.0
        bc_auroc_sum = 0.0
        bc_auprc_sum = 0.0
        for i in range(test_size):  
            y_true = y_test[i].cpu().numpy()  
            x_input = x_test[i].cpu().numpy() 
            auroc, auprc = auroc_auprc(y_true, x_input)
            initial_auroc_sum += auroc
            initial_auprc_sum += auprc
            A_0_nd = ND_regulatory(x_input)
            A_0_bc = BRANE(x_input)
            auroc, auprc = auroc_auprc(y_true, A_0_nd)
            nd_auroc_sum += auroc
            nd_auprc_sum += auprc
            auroc, auprc = auroc_auprc(y_true, A_0_bc)    
            bc_auroc_sum += auroc
            bc_auprc_sum += auprc
        
        initial_auroc = initial_auroc_sum / test_size
        initial_auprc = initial_auprc_sum / test_size
        nd_auroc = nd_auroc_sum / test_size
        nd_auprc = nd_auprc_sum / test_size
        bc_auroc = bc_auroc_sum / test_size
        bc_auprc = bc_auprc_sum / test_size
    
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
        print(f'{initial_auroc:.4f}, {initial_auprc:.4f}, {test_auroc:.4f}, {test_auprc:.4f}, {nd_auroc:.4f}, {nd_auprc:.4f}, {bc_auroc:.4f}, {bc_auprc:.4f}')
    
    
"""
SINC001 = np.array([
    [0.6290, 0.5897],
    [0.7533, 0.6908],
    [0.5227, 0.5431],
    [0.5227, 0.5413],
    [0.4487, 0.5378],
    [0.7844, 0.7189],
    [0.4403, 0.5241],
    [0.4418, 0.5223],
    [0.6493, 0.5726],
    [0.7192, 0.6185],
    [0.5401, 0.5369],
    [0.5422, 0.5375],
    [0.5026, 0.5243],
    [0.5487, 0.5460],
    [0.4971, 0.5205],
    [0.4972, 0.5202]
])

SINC010 = np.array([
    [0.6654, 0.6109],
    [0.8547, 0.7519],
    [0.5131, 0.5447],
    [0.5129, 0.5431],
    [0.4062, 0.5061],
    [0.8703, 0.7672],
    [0.3985, 0.4976],
    [0.3983, 0.4959],
    [0.6783, 0.5829],
    [0.7964, 0.6637],
    [0.5394, 0.5387],
    [0.5450, 0.5409],
    [0.5076, 0.5313],
    [0.5976, 0.5658],
    [0.5053, 0.5251],
    [0.5058, 0.5260]
])

SINC100 = np.array([
    [0.6008, 0.5782],
    [0.6678, 0.6087],
    [0.5045, 0.5314],
    [0.5052, 0.5312],
    [0.3965, 0.4881],
    [0.6757, 0.6040],
    [0.3689, 0.4748],
    [0.3693, 0.4746],
    [0.7154, 0.5967],
    [0.7661, 0.6294],
    [0.5332, 0.5363],
    [0.5374, 0.5381],
    [0.5068, 0.5301],
    [0.5162, 0.5333],
    [0.5073, 0.5236],
    [0.5081, 0.5248]
])
"""
    
