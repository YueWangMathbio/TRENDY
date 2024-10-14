"""
functions of calling each model to infer the GRN
"""

import numpy as np
from previous_methods.wendy_solver import RegRelSolver
from sklearn.covariance import GraphicalLassoCV
import warnings
warnings.filterwarnings("ignore")
import torch
from previous_methods.GENIE3 import GENIE3
import random
from previous_methods.sincerities import sincer
from previous_methods.xgbgrn import get_importances

from models import CovRev, DoubleGRN, TripleGRN


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # If using CUDA
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

 

weight_folder = 'weights'

def wendy_k0kt(k0, kt):
    gene_num = len(k0)
    lam = 0.0 # coefficient of an L2 regularizer, suggested to be 0
    weight = np.ones((gene_num, gene_num)) 
    for i in range(gene_num):
        weight[i, i] = 0.0 # declare that diagonal elements do not count in matching
    solver = RegRelSolver(k0, kt, lam, weight) # call the solver to calculate A
    wendy = solver.fit()
    return wendy



def trendy(data0, data1): # each of data0 and data1 is an m*n matrix, where each row is a cell, and each column is a gene
    set_seed(42) 
    head_num = 4
    layer_num = 7
    model1 = CovRev(n_gene=10, d_model=64, nhead=head_num, num_layers=layer_num, dropout=0.1).to(device)
    model1.load_state_dict(torch.load('weights/trendy_1.pth', map_location=device))
    model1.eval()
    head_num = 8
    layer_num = 7
    model2 = TripleGRN(n_gene=10, d_model=64, nhead=head_num, num_layers=layer_num, dropout=0.1).to(device)
    model2.load_state_dict(torch.load('weights/trendy_2.pth', map_location=device))
    model2.eval()
    
    temp = GraphicalLassoCV().fit(data0)
    k0 = temp.covariance_ 
    temp = GraphicalLassoCV().fit(data1)
    kt = temp.covariance_
    A_0 = wendy_k0kt(k0, kt)
    
    kt = torch.tensor(kt, dtype=torch.float32, device=device).unsqueeze(0)
    ktstar = model1(kt).squeeze(0).cpu().detach().numpy()  
    A_1 = wendy_k0kt(k0, ktstar)
    
    k0 = torch.tensor(k0, dtype=torch.float32, device=device).unsqueeze(0)
    A_1_tensor = torch.tensor(A_1, dtype=torch.float32, device=device).unsqueeze(0)
    A_2 = model2(A_1_tensor, k0, kt).squeeze(0).cpu().detach().numpy()
    
    return A_0, A_2 # inferred GRNs by WENDY and TRENDY


def genie_rev(data): # data is an m*n matrix, where each row is a cell, and each column is a gene
    set_seed(42) 
    head_num = 8
    layer_num = 7
    model = DoubleGRN(n_gene=10, d_model=64, nhead=head_num, num_layers=layer_num, dropout=0.1).to(device)
    model.load_state_dict(torch.load('weights/genie_rev.pth', map_location=device))
    model.eval()
    
    A_0 = GENIE3(data)
    
    temp = GraphicalLassoCV().fit(data)
    k0 = temp.covariance_ 
    k0 = torch.tensor(k0, dtype=torch.float32, device=device).unsqueeze(0)
    A_0_tensor = torch.tensor(A_0, dtype=torch.float32, device=device).unsqueeze(0)
    A_1 = model(A_0_tensor, k0).squeeze(0).cpu().detach().numpy()  
    
    return A_0, A_1 # inferred GRNs by GENIE3 and GENIE3-rev

def sinc_rev(sinc_data, sinc_time): 
    # sinc_data is a list of matrices, each is m*n, where each row is a cell, and each column is a gene
    # sinc_time is a list of time points for the above data
    set_seed(42) 
    head_num = 8
    layer_num = 7
    model = CovRev(n_gene=10, d_model=64, nhead=head_num, num_layers=layer_num, dropout=0.1).to(device)
    model.load_state_dict(torch.load('weights/sinc_rev.pth', map_location=device))
    model.eval()
    A_0 = sincer(sinc_data, sinc_time)
    A_0_tensor = torch.tensor(A_0, dtype=torch.float32, device=device).unsqueeze(0)
    A_1 = model(A_0_tensor).squeeze(0).cpu().detach().numpy()  

    return A_0, A_1 # inferred GRNs by SINCERITIES and SINCERITIES-rev

def nlode_rev(nl_data, nl_time):   
    # nl_data is a list of matrices, each is m*n, where each row is a cell, and each column is a gene
    # nl_time is a list of time points for the above data 
    set_seed(42) 
    head_num = 8
    layer_num = 7
    model = CovRev(n_gene=10, d_model=64, nhead=head_num, num_layers=layer_num, dropout=0.1).to(device)
    model.load_state_dict(torch.load('weights/nlode_rev.pth', map_location=device))
    model.eval()

    bulk_data = np.zeros((2, len(nl_time), nl_data[0].shape[-1]))
    for i in range(len(nl_time)):
        sim_num = nl_data[i].shape[0]
        bulk_data[0, i, :] = np.average(nl_data[i][:sim_num//2, :], axis=0)
        bulk_data[1, i, :] = np.average(nl_data[i][sim_num//2:, :], axis=0)
    nl_time_full = [np.arange(len(nl_time))] * 2
    A_0 = get_importances(bulk_data, nl_time_full, alpha='from_data')
    A_0_tensor = torch.tensor(A_0, dtype=torch.float32, device=device).unsqueeze(0)
    A_1 = model(A_0_tensor).squeeze(0).cpu().detach().numpy()  

    return A_0, A_1 # inferred GRNs by NonlinearODEs and NonlinearODEs-rev


















