from logistic_regression import Logistic_Regression
from Task_Helper import normalize_data, import_data, dump_object, load_object
import os

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
    
    train_data = normalize_data(import_data(train_data_path))[:100000]
    vali_data = normalize_data(import_data(vali_data_path))[:100000]
    test_data = normalize_data(import_data(test_data_path))[:100000]
    
    dump_object(train_data, toy_data_dir + '/train')
    dump_object(vali_data, toy_data_dir + '/vali')
    dump_object(test_data, toy_data_dir + '/test')

#log_reg = Logistic_Regression('L1', eta=0.05)
#log_reg.fit(train_data, r_learn=0.15, iterations=5000, batch_size=1000)

log_reg = Logistic_Regression('L2', eta=0.05)
log_reg.fit(train_data, r_learn=0.2, iterations=5000, batch_size=1000)

test_y = test_data.pop('relevance')
accuracy = log_reg.classification_accuracy(test_data, test_y)
print('Final test accuracy:', accuracy)
