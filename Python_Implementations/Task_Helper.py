import pandas as pd
import numpy as np
import pickle
import re, os

def ndarray_to_pd(array):
    columns = ['relevance', 'qid']
    columns = columns + np.arange(1, 137).tolist()
    return pd.DataFrame(array, columns=columns)

def import_pd(data_path):
    def parse_columns(cell):
        entry = re.sub(r'\w*:', '', cell)
        return float(entry)
    
    columns = ['relevance', 'qid']
    columns = columns + np.arange(1, 137).tolist()
    converters = {column : parse_columns for column in columns}
    
    return pd.read_table(data_path, delim_whitespace=True, header=None, names=columns, converters=converters)

def import_data():
    full_data_dir = '/home/hugo/Documents/IRDM/data_set/Fold1'
    toy_data_dir = '/home/hugo/Documents/IRDM/toy_data'
    
    if os.path.isfile(toy_data_dir + '/train'):
        train_data = load_object(toy_data_dir + '/train')
        vali_data = load_object(toy_data_dir + '/vali')
        test_data = load_object(toy_data_dir + '/test')
        
    else:
        train_data_path = full_data_dir + '/train.txt'
        vali_data_path = full_data_dir + '/vali.txt'
        test_data_path = full_data_dir + '/test.txt'
        
        train_data = import_pd(train_data_path)
        vali_data = import_pd(vali_data_path)
        test_data = import_pd(test_data_path)
        
        dump_object(train_data[:100000], toy_data_dir + '/train')
        dump_object(vali_data[:100000], toy_data_dir + '/vali')
        dump_object(test_data[:100000], toy_data_dir + '/test')
    
    return train_data, vali_data, test_data

def normalize_data(data):
    data_y = data.pop('relevance')
    normalized_data = (data - np.mean(data, axis=0)) / np.std(data, axis=0)
    normalized_data['relevance'] = data_y
    return normalized_data

def numpify(data):
    return np.array(data)

def dump_object(data, save_path):
    with open(save_path, "wb") as handle:
        pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)
        print('Data succesfully dumped at:', save_path)

def load_object(save_path):
    with open(save_path, "rb") as handle:
        object_fetched = pickle.load(handle)
        print('Data succesfully loaded from:', save_path)
    return object_fetched
