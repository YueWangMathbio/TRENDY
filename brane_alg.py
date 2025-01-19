"""
BRANE Cut method for enhancing GRN inference
https://bmcbioinformatics.biomedcentral.com/articles/10.1186/s12859-015-0754-2

Also 
BRANE Clust method for enhancing GRN inference
https://ieeexplore.ieee.org/document/7888506

since they are equivalent when no knowledge of TF is known
"""

import numpy as np

def BRANE(mat):
    mat = np.array(mat)
    mat = np.abs(mat)
    mat = (mat + mat.T) / 2
    return mat