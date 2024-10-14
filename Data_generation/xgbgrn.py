"""
this is the code of the NonlinearODEs method for inferring GRN.
this code is from https://github.com/lab319/GRNs_nonlinear_ODEs.
all printing functions are disabled.
this method is reported in https://academic.oup.com/bioinformatics/article/36/19/4885/5709036
remark: for unknown reasons, this code works on Windows, but not on MacOS.
"""

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning) 
from xgboost import XGBRegressor
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score

def estimate_degradation_rates(TS_data,time_points):
    
    """
    For each gene, the degradation rate is estimated by assuming that the gene expression x(t) follows:
    x(t) =  A exp(-alpha * t) + C_min,
    between the highest and lowest expression values.
    C_min is set to the minimum expression value over all genes and all samples.
    The function is available at the study named dynGENIE3.
    Huynh-Thu, V., Geurts, P. dynGENIE3: dynamical GENIE3 for the inference of
    gene networks from time series expression data. Sci Rep 8, 3384 (2018) doi:10.1038/s41598-018-21715-0
    """
    
    ngenes = TS_data[0].shape[1]
    nexp = len(TS_data)
    
    C_min = TS_data[0].min()
    if nexp > 1:
        for current_timeseries in TS_data[1:]:
            C_min = min(C_min,current_timeseries.min())
    
    alphas = np.zeros((nexp,ngenes))
    
    for (i,current_timeseries) in enumerate(TS_data):
        current_time_points = time_points[i]
        
        for j in range(ngenes):
            
            idx_min = np.argmin(current_timeseries[:,j])
            idx_max = np.argmax(current_timeseries[:,j])
            
            xmin = current_timeseries[idx_min,j]
            xmax = current_timeseries[idx_max,j]
            
            tmin = current_time_points[idx_min]
            tmax = current_time_points[idx_max]
            
            xmin = max(xmin-C_min,1e-6)
            xmax = max(xmax-C_min,1e-6)
                
            xmin = np.log(xmin)
            xmax = np.log(xmax)
            
            alphas[i,j] = (xmax - xmin) / abs(tmin - tmax)
                
    alphas = alphas.max(axis=0)
 
    return alphas       

def get_importances(TS_data, time_points, alpha="from_data"):
    
    ngenes = TS_data[0].shape[1]
    if alpha == "from_data":
        alphas = estimate_degradation_rates(TS_data,time_points)
    else:
        alphas = [alpha] * ngenes
    
    # Get the indices of the candidate regulators
    idx = list(range(ngenes))
    
    # Learn an ensemble of trees for each target gene, and compute scores for candidate regulators
    VIM = np.zeros((ngenes,ngenes))
    
    for i in range(ngenes):
        input_idx = idx.copy()
        if i in input_idx:
            input_idx.remove(i)
        vi = get_importances_single(TS_data, time_points, alphas[i], input_idx, i)
        VIM[i,:] = vi

    return VIM
    


def get_importances_single(TS_data, time_points, alpha, input_idx, output_idx):

    h = 1 # define the value of time step

    ngenes = TS_data[0].shape[1]
    nexp = len(TS_data)
    nsamples_time = sum([expr_data.shape[0] for expr_data in TS_data]) 
    ninputs = len(input_idx)

    # Construct training sample 
    input_matrix_time = np.zeros((nsamples_time-h*nexp,ninputs))
    output_vect_time = np.zeros(nsamples_time-h*nexp)

    nsamples_count = 0
    for (i,current_timeseries) in enumerate(TS_data):
        current_time_points = time_points[i]
        npoints = current_timeseries.shape[0]
        time_diff_current = current_time_points[h:] - current_time_points[:npoints-h]
        current_timeseries_input = current_timeseries[:npoints-h,input_idx]
        current_timeseries_output = (current_timeseries[h:,output_idx] - current_timeseries[:npoints-h,output_idx]) / time_diff_current + alpha*current_timeseries[:npoints-h,output_idx]
        nsamples_current = current_timeseries_input.shape[0]
        input_matrix_time[nsamples_count:nsamples_count+nsamples_current,:] = current_timeseries_input
        output_vect_time[nsamples_count:nsamples_count+nsamples_current] = current_timeseries_output
        nsamples_count += nsamples_current


    input_all = input_matrix_time
    output_all = output_vect_time

    treeEstimator = XGBRegressor()

    # Learn ensemble of trees
    treeEstimator.fit(input_all,output_all)
    
    # Compute importance scores
    feature_importances = treeEstimator.feature_importances_
    vi = np.zeros(ngenes)
    vi[input_idx] = feature_importances
       
    return vi

        
        
        
