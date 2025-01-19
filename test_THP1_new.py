"""
measure the performance of all 16 methods on THP-1 data
"""

from evaluation import auroc_auprc
import numpy as np
import warnings
warnings.filterwarnings("ignore")
from brane_alg import BRANE
from nd_alg import ND_regulatory
from methods import trendy, genie_rev, sinc_rev, nlode_rev

all_time = [0, 1, 6, 12, 24, 48, 72, 96]
A = np.load('THP1/THP1_A.npy')
data = np.load('THP1/THP1_data.npy')

# WENDY and TRENDY
total_auroc0 = 0.0
total_auprc0 = 0.0
total_auroc2 = 0.0
total_auprc2 = 0.0
total_auroc_nd = 0.0
total_auprc_nd = 0.0
total_auroc_bc = 0.0
total_auprc_bc = 0.0
group_count = 28
for wt0 in range(7):
    for wt1 in range(wt0+1, 8):
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
group_count = 8
for gt in range(8):
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
A_0, A_1 = sinc_rev(data, all_time)
total_auroc0, total_auprc0 = auroc_auprc(A, A_0)
total_auroc1, total_auprc1 = auroc_auprc(A, A_1)
A_0_nd = ND_regulatory(A_0)
A_0_bc = BRANE(A_0)
total_auroc_nd, total_auprc_nd = auroc_auprc(A, A_0_nd)
total_auroc_bc, total_auprc_bc = auroc_auprc(A, A_0_bc)   
       
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
A_0, A_1 = nlode_rev(data, all_time)
total_auroc0, total_auprc0 = auroc_auprc(A, A_0)
total_auroc1, total_auprc1 = auroc_auprc(A, A_1)
A_0_nd = ND_regulatory(A_0)
A_0_bc = BRANE(A_0)
total_auroc_nd, total_auprc_nd = auroc_auprc(A, A_0_nd)
total_auroc_bc, total_auprc_bc = auroc_auprc(A, A_0_bc)    
       
print(total_auroc0, total_auprc0, total_auroc1, total_auprc1, \
      total_auroc_nd, total_auprc_nd, total_auroc_bc, total_auprc_bc) 

"""
THP1 = np.array([
    [0.5261, 0.3972],
    [0.5557, 0.3669],
    [0.6112, 0.4203],
    [0.6106, 0.4205],
    [0.4484, 0.3546],
    [0.5506, 0.3781],
    [0.4861, 0.3642],
    [0.4792, 0.3623],
    [0.6261, 0.3852],
    [0.5251, 0.3412],
    [0.5956, 0.3900],
    [0.6067, 0.3798],
    [0.5338, 0.3486],
    [0.4808, 0.3302],
    [0.5521, 0.3482],
    [0.5544, 0.3498]
])
"""