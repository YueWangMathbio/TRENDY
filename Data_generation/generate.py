"""
generate training data for all models
"""

import numpy as np
from wendy_solver import RegRelSolver
from GENIE3 import GENIE3
from sincerities import sincer
from xgbgrn import get_importances
from A_data_generation import A_generation, generate_data
import time
import os
import warnings
warnings.filterwarnings("ignore")
import multiprocessing
from joblib import Parallel, delayed
from sklearn.covariance import GraphicalLassoCV
output_dir = 'total_data_10'
for ds_num in range(100):
    # Zero-pad ds_num to ensure it's always two digits
    ds_str = f"{ds_num:02d}"
    group_num = 1000
    
    n = 10 # number of genes
    prob_p = 0.1 # probability of positive edge
    prob_n = 0.1 # probability of negative edge
    time_points = np.linspace(0.0, 1.0, 11)
    tp_num = len(time_points)
    cell_num = 100
    total_A = np.zeros((group_num, n, n))
    total_data = np.zeros((group_num, tp_num, cell_num, n))
    total_cov = np.zeros((group_num, tp_num, n, n))
    total_revcov = np.zeros((group_num, tp_num-1, n, n))
    total_wendy = np.zeros((group_num, tp_num-1, n, n))
    total_genie = np.zeros((group_num, tp_num-1, n, n))
    total_sinc = np.zeros((group_num, n, n))
    total_nlode = np.zeros((group_num, n, n))
    nl_time = [np.arange(tp_num)] * 2
    time_step = 0.1 
    
    def wendy_direct(k0, kt):
        lam = 0.0 # coefficient of an L2 regularizer, suggested to be 0
        weight = np.ones((n, n)) 
        for i in range(n):
            weight[i, i] = 0.0 # declare that diagonal elements do not count in matching
        solver = RegRelSolver(k0, kt, lam, weight) # call the solver to calculate A
        my_at = solver.fit()
        covdyn = np.around(my_at, decimals=3)
        return covdyn
    
    start_time = time.time() 
    def process_group(counter):
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            A = A_generation(n, prob_p, prob_n)
            data = generate_data(A, n, time_points, cell_num)
            cov_result = np.zeros((tp_num, n, n))
            revcov_result = np.zeros((tp_num-1, n, n))
            wendy_result = np.zeros((tp_num-1, n, n))
            genie_result = np.zeros((tp_num-1, n, n))
            for i in range(tp_num):
                try:
                    temp = GraphicalLassoCV().fit(data[i])
                    cov_result[i] = np.linalg.inv(temp.precision_)
                except:
                    cov_result[i] = np.cov(data[i].T) # covariance matrix at time 0  
                    
            for i in range(1, tp_num):
                revcov_result[i-1] = (np.eye(n)+i*time_step*A.T).dot(cov_result[i]).dot(np.eye(n)+i*time_step*A)
                wendy_result[i-1] = wendy_direct(cov_result[0], cov_result[i])
                genie_result[i-1] = GENIE3(data[i])
        
            sinc_result = sincer(data, time_points)
            
            bulk_data = [
                np.average(data[:, :cell_num//2, :], axis=1),
                np.average(data[:, cell_num//2:, :], axis=1)
            ]
            nlode_result = get_importances(bulk_data, nl_time, alpha='from_data')
            
        return A, data, wendy_result, genie_result, sinc_result, nlode_result, cov_result, revcov_result
    
    def update_results(counter, results):
        total_A[counter] = results[0]
        total_data[counter] = results[1]
        total_wendy[counter] = results[2]
        total_genie[counter] = results[3]
        total_sinc[counter] = results[4]
        total_nlode[counter] = results[5]
        total_cov[counter] = results[6]
        total_revcov[counter] = results[7]
    
    # Use multiprocessing to parallelize the outer loop
    num_cores = multiprocessing.cpu_count()
    
    results = Parallel(n_jobs=num_cores)(delayed(process_group)(counter) for counter in range(group_num))
       
    # Update the total arrays
    for counter, result in enumerate(results):
        update_results(counter, result)
    
    end_time = time.time()
    print(end_time-start_time)
    
    # File prefix with zero-padded ds_num
    file_prefix = os.path.join(output_dir, f"dataset_{ds_str}_")

    # Save all the total_* arrays with the zero-padded numbering in the specified folder
    np.save(f"{file_prefix}total_A.npy", total_A)
    np.save(f"{file_prefix}total_data.npy", total_data)
    np.save(f"{file_prefix}total_wendy.npy", total_wendy)
    np.save(f"{file_prefix}total_genie.npy", total_genie)
    np.save(f"{file_prefix}total_sinc.npy", total_sinc)
    np.save(f"{file_prefix}total_nlode.npy", total_nlode)
    np.save(f"{file_prefix}total_cov.npy", total_cov)
    np.save(f"{file_prefix}total_revcov.npy", total_revcov)

    print(f"All arrays have been saved with the prefix '{file_prefix}' in the folder '{output_dir}'")
    
    
    
    
    
