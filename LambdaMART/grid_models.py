import numpy as np
import logging
import rankpy
from rankpy.queries import Queries
from rankpy.gridsearch import *
from rankpy.queries import find_constant_features
from rankpy.models import LambdaMART
from sklearn.grid_search import ParameterGrid

from ndcg import NDCG
from maprec import MAP
# Turn on logging.
logging.basicConfig(format='%(asctime)s : %(message)s', level=logging.INFO)

# # ... because loading them will be then faster.
training_queries = Queries.load('MSLR-WEB10K/Fold1/training')
validation_queries = Queries.load('MSLR-WEB10K/Fold1/validation')
test_queries = Queries.load('MSLR-WEB10K/Fold1/test')

logging.info('================================================================================')

# Print basic info about query datasets.
logging.info('Train queries: %s' % training_queries)
logging.info('Valid queries: %s' % validation_queries)
logging.info('Test queries: %s' %test_queries)

logging.info('================================================================================')

# Set this to True in order to remove queries containing all documents
# of the same relevance score -- these are useless for LambdaMART.
remove_useless_queries = False

# Find constant query-document features.
cfs = find_constant_features([training_queries, validation_queries, test_queries])

# Get rid of constant features and (possibly) remove useless queries.
training_queries.adjust(remove_features=cfs, purge=remove_useless_queries)
validation_queries.adjust(remove_features=cfs, purge=remove_useless_queries)
test_queries.adjust(remove_features=cfs)

# Print basic info about query datasets.
logging.info('Train queries: %s' % training_queries)
logging.info('Valid queries: %s' % validation_queries)
logging.info('Test queries: %s' % test_queries)

metric = ['nDCG@10']
n_estimators = [1000, 1500]
# max_leaf_nodes = [5, 7, 10]
max_leaf_nodes = [7]
max_features = [0.5, 0.75, None]
# min_samples_split = [2, 4, 6]
min_samples_split = [2]
min_samples_leaf = [25, 50, 100]
learn_rate = [0.1, 0.2, 0.5]
estopping = [10]
n_jobs=[-1]
# random_state=[28]
random_state = [42]

params = {'metric': metric, 'n_estimators': n_estimators, 'max_leaf_nodes':max_leaf_nodes, 'max_features':max_features,
'min_samples_split':min_samples_split, 'min_samples_leaf': min_samples_leaf, 'shrinkage':learn_rate, 'estopping':estopping,
'n_jobs':n_jobs, 'random_state':random_state}
grid_params = ParameterGrid(params)

best_model, params_results = gridsearch(LambdaMART, grid_params, training_queries,
               estopping_queries=None, validation_queries=validation_queries,
               return_models=False, n_jobs=-1, random_state=None)

#output all the params
with open('grid_params_results.txt', 'a') as f:
	for param_set in params_results:
		f.write('%s %s\n' %(str(param_set[0]), str(param_set[2])))

logging.info('================================================================================')

test_ranks = best_model.predict_rankings(test_queries)
dcg_score = NDCG(test_queries, test_ranks).mean_ndcg()
map_score = MAP(test_queries, test_ranks).mean_average_precision()

#evaluate nDCG10 on test queries for best model
logging.info('%s on the test queries: %.8f'
             % ('NDCG@10', dcg_score))

#evaluate MAP on test queries
logging.info('%s on the test queries: %.8f'
             % ('MAP', map_score))

with open('grid_params_results.txt', 'a') as f:
	f.write("BEST MODEL RESULTS \n")
	f.write("nDCG@10 %f\n" %(dcg_score))
	f.write("MAP %f\n" %(map_score))	

#TODO change the save
best_model.save('models/LambdaMART_best_' + str(best_model.n_estimators) + '_' + str(best_model.min_samples_leaf) + '_' + str(best_model.shrinkage))
