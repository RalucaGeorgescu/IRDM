from itertools import combinations as comb
from Task_Helper import ndarray_to_pd
import numpy as np

def generate_pairwise_index_permutations(data):
    return np.array(list(comb(range(len(data)), 2))).astype(int)

def get_pairs(data):
    # produces 2 df's - A and B - of the same size and shape
    # each matching indices between A and B represent a combination of 
    # 2 different entries carrying the same qid

    grouped_data = data.groupby('qid')
    results_by_group = []
    
    for name, group in grouped_data:
        permutations = generate_pairwise_index_permutations(group)
        group = group.as_matrix()
        results = np.take(group, permutations, axis=0)
        results_by_group.append(results)
    
    results_by_group = np.concatenate(results_by_group, axis=0)
    return ndarray_to_pd(results_by_group[:, 0, :]), ndarray_to_pd(results_by_group[:, 1, :])
    