import pandas as pd
import numpy as np
import pickle
import re

def import_data(data_path):
    def parse_columns(cell):
        entry = re.sub(r'\w*:', '', cell)
        return float(entry)
    
    columns = ['relevance', 'qid']
    columns = columns + np.arange(1, 137).tolist()
    converters = {column : parse_columns for column in columns}
    
    return pd.read_table(data_path, delim_whitespace=True, header=None, names=columns, converters=converters)

def normalize_data(data):
    return (data - np.mean(data, axis=0)) / np.std(data, axis=0)

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
