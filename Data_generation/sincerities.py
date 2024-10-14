"""
this is the code of the SINCERITIES method for inferring GRN.
rewritten in Python by Yue Wang, according to https://github.com/CABSEL/SINCERITIES
this method is reported in https://academic.oup.com/bioinformatics/article/34/2/258/4158033
"""
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning) 
import numpy as np
import scipy.stats
from sklearn import linear_model
import pingouin as pg
import pandas as pd

def sincer(data, time_points):
    gene_num = data[0].shape[-1]
    tp_num = len(time_points)

    # used to calculate Spearman partial correlation
    pddata = pd.DataFrame()
    for i in range(gene_num):
        temp = []
        for j in range(tp_num):
            temp = temp + list(data[j][:, i])
        pddata[str(i)] = np.array(temp)
        
    
    # calculate DD (KS distance) for SINCERITIES
    ks_distance = np.zeros((tp_num - 1, gene_num))
    for i in range(tp_num - 1):
        for j in range(gene_num):
            ks_distance[i, j] = scipy.stats.ks_2samp(data[i][:, j], \
                                data[i+1][:, j])[0] / (time_points[i+1] - time_points[i])

    # SINCERITIES algorithm

    sinc_res = np.zeros((gene_num, gene_num))
    for i in range(gene_num):
        reg = linear_model.RidgeCV(alphas=np.logspace(-6, 6, 121))
        reg.fit(ks_distance[:-1, :], ks_distance[1:, i])
        sinc_res[i, :] = reg.coef_
    
    # assign sign from Spearman partial correlation
    for i in range(gene_num):
        for j in range(i+1, gene_num):
            covar = []
            for k in range(gene_num):
                if k != i and k != j:
                    covar.append(str(k))
            sign = pg.partial_corr(data=pddata, x=str(i), y=str(j), covar=covar, method='spearman').iat[0, 1]
            sinc_res[i, j] = abs(sinc_res[i, j]) * sign / abs(sign)
            sinc_res[j, i] = abs(sinc_res[j, i]) * sign / abs(sign)
    for i in range(gene_num):
        sinc_res[i, i] = 0.0
    return sinc_res

