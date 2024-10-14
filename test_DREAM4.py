"""
measure the performance of all eight methods on DREAM4 data
"""

from evaluation import auroc_auprc
import numpy as np
import warnings
warnings.filterwarnings("ignore")

from methods import trendy, genie_rev, sinc_rev, nlode_rev

all_time = list(range(0, 1050, 50))

# WENDY and TRENDY
total_auroc0 = 0.0
total_auprc0 = 0.0
total_auroc2 = 0.0
total_auprc2 = 0.0
group_count = 1050
for wt0 in range(20):
    for wt1 in range(wt0+1, 21):      
        for network in range(5):
            gene_num = 10           
            A = np.load(f'DREAM4/DREAM4_A_{gene_num}_{network}.npy')
            data = np.load(f'DREAM4/DREAM4_data_{gene_num}_{network}.npy')
            A_0, A_2 = trendy(data[wt0], data[wt1])
            aurocp0, auprp0 = auroc_auprc(A, A_0)
            total_auroc0 += aurocp0 / group_count
            total_auprc0 += auprp0 / group_count
            aurocp2, auprp2 = auroc_auprc(A, A_2)    
            total_auroc2 += aurocp2 / group_count
            total_auprc2 += auprp2 / group_count
        
print(total_auroc0, total_auprc0, total_auroc2, total_auprc2)  
# 0.4898831095063243 0.20798115913474335 0.5340686770108194 0.2176740940043071

# GENIE3 and GENIE3-rev
total_auroc0 = 0.0
total_auprc0 = 0.0
total_auroc1 = 0.0
total_auprc1 = 0.0
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
       
print(total_auroc0, total_auprc0, total_auroc1, total_auprc1) 
# 0.5636245191888052 0.22862861339748772 0.4588703026703029 0.17985821735076093

# SINCERITIES and SINCERITIES-rev
total_auroc0 = 0.0
total_auprc0 = 0.0
total_auroc1 = 0.0
total_auprc1 = 0.0
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
       
print(total_auroc0, total_auprc0, total_auroc1, total_auprc1) 
# 0.49084793860248405 0.19185873300364673 0.4994778294869205 0.20338077650776507

# NonlinearODEs and NonlinearODEs-rev
total_auroc0 = 0.0
total_auprc0 = 0.0
total_auroc1 = 0.0
total_auprc1 = 0.0
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
       
print(total_auroc0, total_auprc0, total_auroc1, total_auprc1) 
# 0.4805905796905797 0.1704962026998318 0.5712400464400467 0.24522178423388882