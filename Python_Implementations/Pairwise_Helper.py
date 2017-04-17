from itertools import combinations as comb
from Task_Helper import ndarray_to_pd, dump_object, load_object
import pandas as pd
import numpy as np
import os

pairs_path = '/home/hugo/Documents/IRDM/toy_data/pairs/'

def generate_pairwise_index_permutations(data):
    return np.array(list(comb(range(len(data)), 2))).astype(int)

def get_pairs(data, name):
    # produces 2 df's - A and B - of the same size and shape
    # each matching indices between A and B represent a combination of 
    # 2 different entries carrying the same qid
    # Finally, only pairs with different relevance scores are retained
    
    if os.path.isfile(pairs_path + '/' + name + '_pair_1'):
        pair_1 = load_object(pairs_path + '/' + name + '_pair_1')
        pair_2 = load_object(pairs_path + '/' + name + '_pair_2')
        
    else:
        grouped_data = data.groupby('qid')
        results_by_group = []
        
        for group_name, group in grouped_data:
            permutations = generate_pairwise_index_permutations(group)
            group = group.as_matrix()
            results = np.take(group, permutations, axis=0)
            results_by_group.append(results)
        
        results_by_group = np.concatenate(results_by_group, axis=0)
        pair_1 = ndarray_to_pd(results_by_group[:, 0, :])
        pair_2 = ndarray_to_pd(results_by_group[:, 1, :])
        
        same = pair_1['relevance'] != pair_2['relevance']
        pair_1 = pair_1[same]
        pair_2 = pair_2[same]
        
        dump_object(pair_1, pairs_path + name + '_pair_1')
        dump_object(pair_2, pairs_path + name + '_pair_2')
    
    return pair_1, pair_2
    
def get_pairs_as_batch(data, name, batch_size):
    x_1, x_2 = get_pairs(data, name)
    
    sample_indices = np.random.choice(x_1.shape[0], batch_size, replace=False)
    x_1_batch = x_1.iloc[sample_indices]
    x_2_batch = x_2.iloc[sample_indices]
    y_1_batch = x_1_batch.pop('relevance')
    y_2_batch = x_2_batch.pop('relevance')
    y_batch = pd.concat([y_1_batch, y_2_batch], axis=1)
    return x_1_batch, x_2_batch, y_batch

def get_pairs_as_full_dataset(data, name):
    x_1, x_2 = get_pairs(data, name)
    y_1 = x_1.pop('relevance')
    y_2 = x_2.pop('relevance')
    y = pd.concat([y_1, y_2], axis=1)
    return x_1, x_2, y
