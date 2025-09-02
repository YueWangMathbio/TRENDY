"""
measure the performance of all 16 methods on mESC data
"""

from evaluation import auroc_auprc
import numpy as np
import warnings
warnings.filterwarnings("ignore")
from brane_alg import BRANE
from nd_alg import ND_regulatory
from methods import trendy, genie_rev, sinc_rev, nlode_rev

all_time = [0, 12, 24, 48, 72]
data = []
for i in range(5):   
    temp = np.load(f'mESC/mESC_new_data{i}.npy')
    data.append(temp)
A = np.load('mESC/mESC_new_A.npy')

# WENDY and TRENDY
total_auroc0 = 0.0
total_auprc0 = 0.0
total_auroc2 = 0.0
total_auprc2 = 0.0
total_auroc_nd = 0.0
total_auprc_nd = 0.0
total_auroc_bc = 0.0
total_auprc_bc = 0.0
group_count = 10
for wt0 in range(4):
    for wt1 in range(wt0+1, 5):
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
res = [total_auroc0, total_auprc0, total_auroc2, total_auprc2, \
      total_auroc_nd, total_auprc_nd, total_auroc_bc, total_auprc_bc]  
np.save('temp_r1.npy', res)      
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
group_count = 5
for gt in range(5):
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
res = [total_auroc0, total_auprc0, total_auroc1, total_auprc1, \
      total_auroc_nd, total_auprc_nd, total_auroc_bc, total_auprc_bc]  
np.save('temp_r2.npy', res)        
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
res = [total_auroc0, total_auprc0, total_auroc1, total_auprc1, \
      total_auroc_nd, total_auprc_nd, total_auroc_bc, total_auprc_bc]  
np.save('temp_r3.npy', res)        
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
res = [total_auroc0, total_auprc0, total_auroc1, total_auprc1, \
      total_auroc_nd, total_auprc_nd, total_auroc_bc, total_auprc_bc]  
np.save('temp_r4.npy', res)        
print(total_auroc0, total_auprc0, total_auroc1, total_auprc1, \
      total_auroc_nd, total_auprc_nd, total_auroc_bc, total_auprc_bc) 


"""
mESC = [[0.4857 0.0411]
 [0.4655 0.0489]
 [0.4273 0.037 ]
 [0.4296 0.037 ]
 [0.5024 0.0452]
 [0.5401 0.0556]
 [0.4779 0.0424]
 [0.473  0.0422]
 [0.5744 0.063 ]
 [0.493  0.0411]
 [0.5896 0.0542]
 [0.5755 0.0517]
 [0.494  0.0517]
 [0.3737 0.0323]
 [0.4957 0.0518]
 [0.4955 0.0514]]
"""



