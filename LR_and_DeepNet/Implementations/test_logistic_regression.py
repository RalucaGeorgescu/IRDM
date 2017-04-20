from logistic_regression import Logistic_Regression
from Task_Helper import import_data, normalize_data
import numpy as np

train_data, vali_data, test_data = import_data()

train_data = normalize_data(train_data)
vali_data = normalize_data(vali_data)
test_data = normalize_data(test_data)

# add a constant column so that we don't need to implement the bias term in the models
train_data['const'] = np.ones(train_data.shape[0])
test_data['const'] = np.ones(test_data.shape[0])
vali_data['const'] = np.ones(vali_data.shape[0])

vali_y = vali_data.pop('relevance')
test_y = test_data.pop('relevance')

# MAP: 0.4728
# NDCG: 0.1911
# Accuracy: 0.4757
log_reg = Logistic_Regression('L1', eta=0.05)
log_reg.fit(train_data, r_learn=0.15, iterations=5000, batch_size=1000)

accuracy = log_reg.classification_accuracy(test_data, test_y)
map_score, ndcg_score = log_reg.get_MAP_and_NDCG(test_data, test_y)
print('Final test accuracy:', accuracy)
print('Final MAP score:', map_score)
print('Final NDCG score:', ndcg_score)

# MAP: 0.4715
# NDCG: 0.1919
# Accuracy: 0.4228
log_reg = Logistic_Regression('L1', eta=0.01)
log_reg.fit(train_data, r_learn=0.1, iterations=5000, batch_size=1000)

accuracy = log_reg.classification_accuracy(test_data, test_y)
map_score, ndcg_score = log_reg.get_MAP_and_NDCG(test_data, test_y)
print('Final test accuracy:', accuracy)
print('Final MAP score:', map_score)
print('Final NDCG score:', ndcg_score)

# MAP: 0.4737
# NDCG: 0.1928
# Accuracy: 0.5523
log_reg = Logistic_Regression(reg='L2', eta=0.5)
log_reg.fit(train_data, r_learn=0.5, iterations=15000, batch_size=1000)

accuracy = log_reg.classification_accuracy(test_data, test_y)
map_score, ndcg_score = log_reg.get_MAP_and_NDCG(test_data, test_y)
print('Final test accuracy:', accuracy)
print('Final MAP score:', map_score)
print('Final NDCG score:', ndcg_score)

# MAP: 0.4738
# NDCG: 0.1929
# Accuracy: 0.5531
log_reg = Logistic_Regression(reg='L2', eta=1.5)
log_reg.fit(train_data, r_learn=2.0, iterations=15000, batch_size=1000)

accuracy = log_reg.classification_accuracy(test_data, test_y)
map_score, ndcg_score = log_reg.get_MAP_and_NDCG(test_data, test_y)
print('Final test accuracy:', accuracy)
print('Final MAP score:', map_score)
print('Final NDCG score:', ndcg_score)

