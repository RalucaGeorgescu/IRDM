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

models = []
vali_scores = []

regularizations = np.power(10.0, np.arange(-10, 10))
r_learns = np.power(10.0, np.arange(-4, 1))

vali_y = vali_data.pop('relevance')

#log_reg = Logistic_Regression('L1', eta=0.05)
#log_reg.fit(train_data, r_learn=0.15, iterations=5000, batch_size=1000)

#log_reg = Logistic_Regression('L1', eta=0.01)
#log_reg.fit(train_data, r_learn=0.1, iterations=5000, batch_size=1000)
log_reg = Logistic_Regression(reg='L2', eta=1.5)
log_reg.fit(train_data, r_learn=2.0, iterations=15000, batch_size=1000)

test_y = test_data.pop('relevance')
accuracy = log_reg.classification_accuracy(test_data, test_y)
print('Final test accuracy:', accuracy)
