import pandas as pd
import numpy as np
import tensorflow as tf
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
    full_data_dir = '../All'
    toy_data_dir = '../processed_data'
    
    if os.path.isfile(toy_data_dir + '/train'):
        train_data = load_object(toy_data_dir + '/train')
        vali_data = load_object(toy_data_dir + '/vali')
        test_data = load_object(toy_data_dir + '/test')
        
    else:
        train_data_path = full_data_dir + '/train.txt'
        vali_data_path = full_data_dir + '/vali.txt'
        test_data_path = full_data_dir + '/test.txt'
        
        print('Processing training set...')
        train_data = import_pd(train_data_path)
        print('Processing validation set...')
        vali_data = import_pd(vali_data_path)
        print('Processing test set...')
        test_data = import_pd(test_data_path)
        
        dump_object(train_data, toy_data_dir + '/train')
        dump_object(vali_data, toy_data_dir + '/vali')
        dump_object(test_data, toy_data_dir + '/test')
    
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

def load_object(save_path):
    with open(save_path, "rb") as handle:
        object_fetched = pickle.load(handle)
    return object_fetched

def load_tf_model(session, model_dir, model_filename):
    model_path = model_dir + "/" + model_filename
    saver = tf.train.Saver()
    saver.restore(session, model_path)
    print('Model succesfully loaded from: ', model_path)

def save_tf_model(sess, model_dir, model_filename):
    if not os.path.exists(model_dir):
        os.mkdir(model_dir)

    model_path = model_dir + "/" + model_filename
    saver = tf.train.Saver()
    saver.save(sess, model_path)
    print('Model succesfully saved at:', model_path)
