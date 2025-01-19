"""
measure the performance of all 16 methods on DREAM4 data
"""

from evaluation import auroc_auprc
import numpy as np
import warnings
warnings.filterwarnings("ignore")
from brane_alg import BRANE
from nd_alg import ND_regulatory
from methods import trendy, genie_rev, sinc_rev, nlode_rev

all_time = list(range(0, 1050, 50))

# WENDY and TRENDY
total_auroc0 = 0.0
total_auprc0 = 0.0
total_auroc2 = 0.0
total_auprc2 = 0.0
total_auroc_nd = 0.0
total_auprc_nd = 0.0
total_auroc_bc = 0.0
total_auprc_bc = 0.0
group_count = 1050
for wt0 in range(20):
    for wt1 in range(wt0+1, 21):      
        for network in range(5):
            gene_num = 10           
            A = np.load(f'DREAM4/DREAM4_A_{gene_num}_{network}.npy')
            data = np.load(f'DREAM4/DREAM4_data_{gene_num}_{network}.npy')            
            A_0, A_2 = trendy(data[wt0], data[wt1])
            auroc, auprc = auroc_auprc(A, A_0)
            total_auroc0 += auroc / group_count
            total_auprc0 += auprc / group_count
            auroc, auprc = auroc_auprc(A, A_2)    
            total_auroc2 += auroc / group_count
            total_auprc2 += auprc / group_count
            A_0_nd = ND_regulatory(A_0)
            A_0_bc = BRANE(A_0)
            auroc, auprc = auroc_auprc(A, A_0_nd)
            total_auroc_nd += auroc / group_count
            total_auprc_nd += auprc / group_count
            auroc, auprc = auroc_auprc(A, A_0_bc)    
            total_auroc_bc += auroc / group_count
            total_auprc_bc += auprc / group_count
        
print(total_auroc0, total_auprc0, total_auroc2, total_auprc2, \
      total_auroc_nd, total_auprc_nd, total_auroc_bc, total_auprc_bc) 

# GENIE3 and GENIE3-rev
total_auroc0 = 0.0
total_auprc0 = 0.0
total_auroc1 = 0.0
total_auprc1 = 0.0
total_auroc_nd = 0.0
total_auprc_nd = 0.0
total_auroc_bc = 0.0
total_auprc_bc = 0.0
group_count = 105
for gt in range(21):
    for network in range(5):
        gene_num = 10           
        A = np.load(f'DREAM4/DREAM4_A_{gene_num}_{network}.npy')
        data = np.load(f'DREAM4/DREAM4_data_{gene_num}_{network}.npy')
        A_0, A_1 = genie_rev(data[gt])
        aurocp0, auprp0 = auroc_auprc(A, A_0)
        total_auroc0 += aurocp0 / group_count
        total_auprc0 += auprp0 / group_count
        aurocp1, auprp1 = auroc_auprc(A, A_1)
        total_auroc1 += aurocp1 / group_count
        total_auprc1 += auprp1 / group_count
        A_0_nd = ND_regulatory(A_0)
        A_0_bc = BRANE(A_0)
        auroc, auprc = auroc_auprc(A, A_0_nd)
        total_auroc_nd += auroc / group_count
        total_auprc_nd += auprc / group_count
        auroc, auprc = auroc_auprc(A, A_0_bc)    
        total_auroc_bc += auroc / group_count
        total_auprc_bc += auprc / group_count
           
print(total_auroc0, total_auprc0, total_auroc1, total_auprc1, \
          total_auroc_nd, total_auprc_nd, total_auroc_bc, total_auprc_bc) 

# SINCERITIES and SINCERITIES-rev
total_auroc0 = 0.0
total_auprc0 = 0.0
total_auroc1 = 0.0
total_auprc1 = 0.0
total_auroc_nd = 0.0
total_auprc_nd = 0.0
total_auroc_bc = 0.0
total_auprc_bc = 0.0
group_count = 55
for st in range(11):
    for network in range(5):
        gene_num = 10           
        A = np.load(f'DREAM4/DREAM4_A_{gene_num}_{network}.npy')
        data = np.load(f'DREAM4/DREAM4_data_{gene_num}_{network}.npy')
        sinc_data = data[st: st+11]
        sinc_time = all_time[st: st+11]
        A_0, A_1 = sinc_rev(sinc_data, sinc_time)
        aurocp0, auprp0 = auroc_auprc(A, A_0)
        total_auroc0 += aurocp0 / group_count
        total_auprc0 += auprp0 / group_count
        aurocp1, auprp1 = auroc_auprc(A, A_1)
        total_auroc1 += aurocp1 / group_count
        total_auprc1 += auprp1 / group_count
        A_0_nd = ND_regulatory(A_0)
        A_0_bc = BRANE(A_0)
        auroc, auprc = auroc_auprc(A, A_0_nd)
        total_auroc_nd += auroc / group_count
        total_auprc_nd += auprc / group_count
        auroc, auprc = auroc_auprc(A, A_0_bc)    
        total_auroc_bc += auroc / group_count
        total_auprc_bc += auprc / group_count
       
print(total_auroc0, total_auprc0, total_auroc1, total_auprc1, \
          total_auroc_nd, total_auprc_nd, total_auroc_bc, total_auprc_bc) 

# NonlinearODEs and NonlinearODEs-rev
total_auroc0 = 0.0
total_auprc0 = 0.0
total_auroc1 = 0.0
total_auprc1 = 0.0
total_auroc_nd = 0.0
total_auprc_nd = 0.0
total_auroc_bc = 0.0
total_auprc_bc = 0.0
group_count = 55
for nt in range(11):
    for network in range(5):
        gene_num = 10           
        A = np.load(f'DREAM4/DREAM4_A_{gene_num}_{network}.npy')
        data = np.load(f'DREAM4/DREAM4_data_{gene_num}_{network}.npy')
        nl_data = data[st: st+11]
        nl_time = all_time[st: st+11]
        A_0, A_1 = nlode_rev(nl_data, nl_time)
        aurocp0, auprp0 = auroc_auprc(A, A_0)
        total_auroc0 += aurocp0 / group_count
        total_auprc0 += auprp0 / group_count
        aurocp1, auprp1 = auroc_auprc(A, A_1)
        total_auroc1 += aurocp1 / group_count
        total_auprc1 += auprp1 / group_count
        A_0_nd = ND_regulatory(A_0)
        A_0_bc = BRANE(A_0)
        auroc, auprc = auroc_auprc(A, A_0_nd)
        total_auroc_nd += auroc / group_count
        total_auprc_nd += auprc / group_count
        auroc, auprc = auroc_auprc(A, A_0_bc)    
        total_auroc_bc += auroc / group_count
        total_auprc_bc += auprc / group_count
       
print(total_auroc0, total_auprc0, total_auroc1, total_auprc1, \
          total_auroc_nd, total_auprc_nd, total_auroc_bc, total_auprc_bc) 


"""
DREAM4 = np.array([
    [0.4899, 0.2080],
    [0.5341, 0.2177],
    [0.5417, 0.2254],
    [0.5421, 0.2231],
    [0.5636, 0.2286],
    [0.4589, 0.1799],
    [0.5632, 0.2261],
    [0.5741, 0.2284],
    [0.4908, 0.1919],
    [0.4995, 0.2034],
    [0.4999, 0.1856],
    [0.5040, 0.1846],
    [0.4806, 0.1705],
    [0.5712, 0.2452],
    [0.4791, 0.1772],
    [0.4856, 0.1666]
])
"""