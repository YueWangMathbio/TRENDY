"""
network deconvolution method for enhancing GRN inference
https://www.nature.com/articles/nbt.2635
"""

import numpy as np

def ND_regulatory(mat):
    mat = np.array(mat)
    beta = 0.5
    alpha = 0.1
    mat = np.abs(mat)
    n, _ = mat.shape
    np.fill_diagonal(mat, 0)
    mat = np.maximum(mat, mat.T)
    if np.all(mat == mat[0, 0]):
        return mat

    threshold = np.quantile(mat.flatten(), 1 - alpha)
    mat_th = np.where(mat >= threshold, mat, 0)
    mat_th = (mat_th + mat_th.T) / 2
    
    temp_net = (mat_th > 0).astype(float)
    temp_net_remain = (mat_th == 0).astype(float)
    mat_th_remain = mat * temp_net_remain
    m11 = np.max(mat_th_remain)

    D, U = np.linalg.eigh(mat_th)
    D = np.diag(D)
    if np.linalg.cond(U) > 1e10:
        r_p = 0.001
        np.random.seed(1)
        mat_rand = r_p * np.random.rand(n, n)
        mat_rand = (mat_rand + mat_rand.T) / 2
        np.fill_diagonal(mat_rand, 0)
        mat_th += mat_rand
        mat_th = (mat_th + mat_th.T) / 2
        D, U = np.linalg.eigh(mat_th)
        D = np.diag(D)

    lam_n = abs(min(D.min(), 0))
    lam_p = abs(max(D.max(), 0))
    m1 = lam_p * (1 - beta) / beta
    m2 = lam_n * (1 + beta) / beta
    scale_eigen = max(m1, m2)

    for i in range(n):
        D[i] = D[i] / (scale_eigen + D[i])

    net_new = U @ D @ np.linalg.inv(U)
    m2 = net_new.min()
    net_new3 = (net_new + max(m11 - m2, 0)) * temp_net
    mat_nd = net_new3 + mat_th_remain
    return mat_nd
