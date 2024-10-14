
"""
# the following function calculate auroc and aupr to evaluate
# the calculated GRN and the true GRN.
# the input A is the true GRN (has only 0, 1, -1), 
# and the input result is the calculated GRN
# this function requires matching both the direction of the regulation
# and the activation/inhibition of the regulation

# see the SINCERETIES paper Section 3.1 for more details
# https://academic.oup.com/bioinformatics/article/34/2/258/4158033
"""
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score

def auroc_auprc(A, result): # A is true GRN, result is inferred GRN
    n = A.shape[0]
    
    pairwise_aurocs = []
    pairwise_auprcs = []
    total_weights = 0
    
    pairs = [(1, -1), (1, 0), (0, -1)]
    
    for pos_label, neg_label in pairs:
        # Filter data for the current pair, excluding diagonal elements
        mask = ((A == pos_label) | (A == neg_label)) & (np.eye(n) == 0)
        
        binary_y_true = A[mask]
        binary_y_pred = result[mask]
        
        # Convert labels to binary (1 vs 0)
        binary_y_true = (binary_y_true == pos_label).astype(int)
        
        # Calculate the weight: number of pos_label * number of neg_label
        pos_count = binary_y_true.sum()
        neg_count = len(binary_y_true) - pos_count
        weight = pos_count * neg_count
        
        if weight > 0:  # Avoid calculations if no valid pairs
            # Compute AUROC and AUPRC
            auroc = roc_auc_score(binary_y_true, binary_y_pred)
            auprc = average_precision_score(binary_y_true, binary_y_pred)
            
            # Store the weighted results
            pairwise_aurocs.append(auroc * weight)
            pairwise_auprcs.append(auprc * weight)
            total_weights += weight
    
    # Calculate the weighted average
    avg_auroc = sum(pairwise_aurocs) / total_weights if total_weights > 0 else 0
    avg_auprc = sum(pairwise_auprcs) / total_weights if total_weights > 0 else 0
    
    return avg_auroc, avg_auprc
