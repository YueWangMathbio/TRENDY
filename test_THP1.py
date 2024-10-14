"""
measure the performance of all eight methods on THP-1 data
"""

from evaluation import auroc_auprc
import numpy as np
import warnings
warnings.filterwarnings("ignore")

from methods import trendy, genie_rev, sinc_rev, nlode_rev

all_time = [0, 1, 6, 12, 24, 48, 72, 96]
A = np.load('THP1/THP1_A.npy')
data = np.load('THP1/THP1_data.npy')

# WENDY and TRENDY
total_auroc0 = 0.0
total_auprc0 = 0.0
total_auroc2 = 0.0
total_auprc2 = 0.0
group_count = 28
for wt0 in range(7):
    for wt1 in range(wt0+1, 8):
        A_0, A_2 = trendy(data[wt0], data[wt1])
        aurocp0, auprp0 = auroc_auprc(A, A_0)
        total_auroc0 += aurocp0 / group_count
        total_auprc0 += auprp0 / group_count
        aurocp2, auprp2 = auroc_auprc(A, A_2)    
        total_auroc2 += aurocp2 / group_count
        total_auprc2 += auprp2 / group_count
        
print(total_auroc0, total_auprc0, total_auroc2, total_auprc2)        
# 0.5261111327453156 0.39720727019859325 0.5556775183836653 0.3668694743656891

# GENIE3 and GENIE3-rev
total_auroc0 = 0.0
total_auprc0 = 0.0
total_auroc1 = 0.0
total_auprc1 = 0.0
group_count = 8
for gt in range(8):
    A_0, A_1 = genie_rev(data[gt])
    aurocp0, auprp0 = auroc_auprc(A, A_0)
    total_auroc0 += aurocp0 / group_count
    total_auprc0 += auprp0 / group_count
    aurocp1, auprp1 = auroc_auprc(A, A_1)
    total_auroc1 += aurocp1 / group_count
    total_auprc1 += auprp1 / group_count
       
print(total_auroc0, total_auprc0, total_auroc1, total_auprc1) 
# 0.4484178365362773 0.35461941018748144 0.5505826632138475 0.3780519519746344

# SINCERITIES and SINCERITIES-rev
total_auroc0 = 0.0
total_auprc0 = 0.0
total_auroc1 = 0.0
total_auprc1 = 0.0
A_0, A_1 = sinc_rev(data, all_time)
total_auroc0, total_auprc0 = auroc_auprc(A, A_0)
total_auroc1, total_auprc1 = auroc_auprc(A, A_1)
       
print(total_auroc0, total_auprc0, total_auroc1, total_auprc1) 
# 0.6260960428876472 0.3851944345282084 0.5251465176502658 0.3411664571234295

# NonlinearODEs and NonlinearODEs-rev
total_auroc0 = 0.0
total_auprc0 = 0.0
total_auroc1 = 0.0
total_auprc1 = 0.0
A_0, A_1 = nlode_rev(data, all_time)
total_auroc0, total_auprc0 = auroc_auprc(A, A_0)
total_auroc1, total_auprc1 = auroc_auprc(A, A_1)
       
print(total_auroc0, total_auprc0, total_auroc1, total_auprc1) 
# 0.5337558493480532 0.3486250587687994 0.48080505201944485 0.3302202692676213

