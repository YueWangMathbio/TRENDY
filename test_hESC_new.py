"""
measure the performance of all 16 methods on hESC data
"""

from evaluation import auroc_auprc
import numpy as np
import warnings
warnings.filterwarnings("ignore")
from brane_alg import BRANE
from nd_alg import ND_regulatory
from methods import trendy, genie_rev, sinc_rev, nlode_rev

all_time = [0, 12, 24, 36, 72, 96]
data = []
for i in range(6):   
    temp = np.load(f'hESC/hESC_data{i}.npy')
    data.append(temp)
A = np.load('hESC/hESC_A.npy')

# WENDY and TRENDY
total_auroc0 = 0.0
total_auprc0 = 0.0
total_auroc2 = 0.0
total_auprc2 = 0.0
total_auroc_nd = 0.0
total_auprc_nd = 0.0
total_auroc_bc = 0.0
total_auprc_bc = 0.0
group_count = 15
for wt0 in range(5):
    for wt1 in range(wt0+1, 6):
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
group_count = 6
for gt in range(6):
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
hESC = np.array([
    [0.4997, 0.0392],
    [0.5311, 0.0376],
    [0.4971, 0.0372],
    [0.5070, 0.0402],
    [0.5913, 0.0468],
    [0.6008, 0.0435],
    [0.5744, 0.0462],
    [0.5767, 0.0488],
    [0.4198, 0.0261],
    [0.4871, 0.0294],
    [0.1955, 0.0199],
    [0.1842, 0.0196],
    [0.5971, 0.0534],
    [0.6233, 0.0641],
    [0.6008, 0.0466],
    [0.6040, 0.0633]
])
"""



