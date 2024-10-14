"""
measure the performance of all eight methods on hESC data
"""

from evaluation import auroc_auprc
import numpy as np
import warnings
warnings.filterwarnings("ignore")
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
group_count = 15
for wt0 in range(5):
    for wt1 in range(wt0+1, 6):
        A_0, A_2 = trendy(data[wt0], data[wt1])
        aurocp0, auprp0 = auroc_auprc(A, A_0)
        total_auroc0 += aurocp0 / group_count
        total_auprc0 += auprp0 / group_count
        aurocp2, auprp2 = auroc_auprc(A, A_2)    
        total_auroc2 += aurocp2 / group_count
        total_auprc2 += auprp2 / group_count
        
print(total_auroc0, total_auprc0, total_auroc2, total_auprc2)
# 0.4997381219603442 0.03923750540614212 0.5311136051876791 0.03762202591092596

# GENIE3 and GENIE3-rev
total_auroc0 = 0.0
total_auprc0 = 0.0
total_auroc1 = 0.0
total_auprc1 = 0.0
group_count = 6
for gt in range(6):
    A_0, A_1 = genie_rev(data[gt])
    aurocp0, auprp0 = auroc_auprc(A, A_0)
    total_auroc0 += aurocp0 / group_count
    total_auprc0 += auprp0 / group_count
    aurocp1, auprp1 = auroc_auprc(A, A_1)
    total_auroc1 += aurocp1 / group_count
    total_auprc1 += auprp1 / group_count
       
print(total_auroc0, total_auprc0, total_auroc1, total_auprc1) 
# 0.5913455543085173 0.04679668459239553 0.6008230452674896 0.04352233002621763

# SINCERITIES and SINCERITIES-rev
total_auroc0 = 0.0
total_auprc0 = 0.0
total_auroc1 = 0.0
total_auprc1 = 0.0
A_0, A_1 = sinc_rev(data, all_time)
total_auroc0, total_auprc0 = auroc_auprc(A, A_0)
total_auroc1, total_auprc1 = auroc_auprc(A, A_1)
       
print(total_auroc0, total_auprc0, total_auroc1, total_auprc1) 
# 0.41975308641975306 0.026115905536391888 0.48709315375982043 0.029416876353647394

# NonlinearODEs and NonlinearODEs-rev
total_auroc0 = 0.0
total_auprc0 = 0.0
total_auroc1 = 0.0
total_auprc1 = 0.0
A_0, A_1 = nlode_rev(data, all_time)
total_auroc0, total_auprc0 = auroc_auprc(A, A_0)
total_auroc1, total_auprc1 = auroc_auprc(A, A_1)
       
print(total_auroc0, total_auprc0, total_auroc1, total_auprc1) 
# 0.5970819304152637 0.05343394932602222 0.6232697343808454 0.06410540413448676






