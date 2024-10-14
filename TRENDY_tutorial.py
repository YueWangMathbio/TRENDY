"""
This file explains how to use TRENDY method to calculate the gene regulatory
network from single-cell level gene expression data, measured at two points 
after some general interventions, where the joint distribution of these two
time points is unknown.
"""

import numpy as np
from previous_methods.wendy_solver import RegRelSolver
from sklearn.covariance import GraphicalLassoCV
import warnings
warnings.filterwarnings("ignore")
import torch
from models import CovRev, TripleGRN


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
    head_num = 4
    layer_num = 7
    model1 = CovRev(n_gene=10, d_model=64, nhead=head_num, num_layers=layer_num, dropout=0.1).to(device)
    model1.load_state_dict(torch.load('weights/trendy_1.pth', map_location=device))
    model1.eval()
    # TE(k=1) part of TRENDY
    
    head_num = 8
    layer_num = 7
    model2 = TripleGRN(n_gene=10, d_model=64, nhead=head_num, num_layers=layer_num, dropout=0.1).to(device)
    model2.load_state_dict(torch.load('weights/trendy_2.pth', map_location=device))
    model2.eval()
    # TE(k=3) part of TRENDY
    
    temp = GraphicalLassoCV().fit(data0)
    k0 = temp.covariance_ # covariance matrix of data0
    temp = GraphicalLassoCV().fit(data1)
    kt = temp.covariance_ # covariance matrix of data0
    A_0 = wendy_k0kt(k0, kt) # inferred GRN by WENDY
    
    kt = torch.tensor(kt, dtype=torch.float32, device=device).unsqueeze(0)
    ktstar = model1(kt).squeeze(0).cpu().detach().numpy() # use TE(k=1) model to calculate Kt'
    A_1 = wendy_k0kt(k0, ktstar) # inferred GRN by the first half of TRENDY
    
    k0 = torch.tensor(k0, dtype=torch.float32, device=device).unsqueeze(0)
    A_1_tensor = torch.tensor(A_1, dtype=torch.float32, device=device).unsqueeze(0)
    A_2 = model2(A_1_tensor, k0, kt).squeeze(0).cpu().detach().numpy() 
    # second half of TRENDY, use K0, Kt, A_1 in TE(k=3) model to infer the final GRN
    
    return A_0, A_2 # inferred GRNs by WENDY and TRENDY



"""
if you have raw single-cell RNA sequencing (scRNAseq) data:
use scanpy or other packages to extract expression data at each time point.
remove genes that only appear in a few cells, and cells that only a few genes
are measured.
replace each value x by log(1+x).
for each cell (row), normalize its sum, so that each cell has the same 
total expression level. 
"""

# this is an example of using WENDY
data0 = np.load('example_data_0.npy')
data1 = np.load('example_data_1.npy')
# here are two data sets, each is a numpy array of size 100 * 10,
# meaning the expression levels of 10 genes for 100 cells 


grn_wendy, grn_trendy = trendy(data0, data1) # this is the calculate GRN.
# grn[i, j] means the regulation strength of gene i on gene j
# positive means activation, negative means inhibition.

print(grn_wendy, grn_trendy)
