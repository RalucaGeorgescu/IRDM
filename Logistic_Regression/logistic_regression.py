from Task_Helper import numpify
import numpy as np

n_relevance = 5
n_attributes = 137

class Logistic_Regression(object):
    
    def __init__(self, reg, eta):
        self.reg = reg
        self.eta = eta
        self.W = np.zeros([n_attributes, n_relevance], dtype=float)
    
    def fit(self, data, r_learn, iterations=10000, batch_size=10000):
        for i in range(iterations):
            batch_x, batch_y = self.__get_batch(data, batch_size)
            self.W = self.W - r_learn * self.__compute_gradients(batch_x, batch_y)
            
            if i % 100 == 0:
                print('Iteration: {} Regularized log-likelihood: {}'.format(i, self.__reg_log_likelihood(batch_x, batch_y)))
                print('Least sq error:', self.__least_sq_error(batch_x, batch_y))
    
    def classification_accuracy(self, test_x, test_y):
        test_x = numpify(test_x).reshape(-1, n_attributes)
        test_y = numpify(test_y).reshape(-1, 1)
        predictions = np.argmax(self.__softmax(self.W, test_x), axis=1)
        return np.mean(test_y == predictions)
        
    def __softmax(self, W, x):
        numerator = np.exp(np.dot(x, W))
        return numerator / np.sum(numerator, axis=0)

    def __reg_log_likelihood(self, x, y):
        log_probabilities = np.log(self.__softmax(self.W, x))
        indicators = (y == np.array([0, 1, 2, 3, 4])).astype(int)
        
        if self.reg == 'L1':
            regularization = self.eta * np.sum(np.abs(self.W))
        elif self.reg == 'L2':
            regularization = 0.5 * self.eta * np.sum(self.W ** 2)
        
        return -np.sum(np.multiply(indicators, log_probabilities)) + regularization

    def __compute_gradients(self, x, y):
        probabilities = self.__softmax(self.W, x)
        indicators = (y == np.array([0, 1, 2, 3, 4])).astype(int)
        
        if self.reg == 'L1':
            regularization = self.eta * np.sign(self.W)
        elif self.reg == 'L2':
            regularization = self.eta * self.W
            
        return -np.sum(x.T[:, :, None] * (indicators - probabilities), axis=1) + regularization
        
    def __get_batch(self, data, batch_size):
        batch_x = data.sample(n=batch_size, replace=False)
        batch_y = batch_x.pop('relevance')
        batch_x = numpify(batch_x).reshape(batch_size, -1)
        batch_y = numpify(batch_y).reshape(batch_size, -1)
        return batch_x, batch_y

    def __least_sq_error(self, test_x, test_y):
        predictions = np.argmax(self.__softmax(self.W, test_x), axis=1)
        return np.mean((test_y - predictions) ** 2)    
    