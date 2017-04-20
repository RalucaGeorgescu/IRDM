import numpy as np
import logging
import rankpy
from rankpy.queries import Queries
from rankpy.queries import find_constant_features
from rankpy.models import LambdaMART
from sklearn.grid_search import ParameterGrid

from ndcg import NDCG
from maprec import MAP

# Turn on logging.
logging.basicConfig(format='%(asctime)s : %(message)s', level=logging.INFO)
dcg_folds_scores = []
map_folds_scores = []
# load the data for each fold
for i in xrange(1, 6):	
	# load training, validation and testing sets for current
	training_queries = Queries.load('MSLR-WEB10K/Fold' + str(i) + '/training')
	validation_queries = Queries.load('MSLR-WEB10K/Fold' + str(i) + '/validation')
	test_queries = Queries.load('MSLR-WEB10K/Fold' + str(i) + '/test')

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

	logging.info('================================================================================')	

	#parameters
	metric = 'nDCG@10'
	max_leaf_nodes = 7
	min_samples_split = 2
	estopping = 10	
	#TODO change these to the optimal ones found from the grid search
	n_estimators = 1000
	max_features = None
	min_samples_leaf = 50
	learn_rate = 0.2

	model = LambdaMART(metric=metric, max_leaf_nodes=max_leaf_nodes, min_samples_split=min_samples_split, 
                   estopping=estopping, n_estimators=n_estimators, max_features=max_features, min_samples_leaf=min_samples_leaf, 
                   shrinkage=learn_rate, n_jobs=-1, random_state=42)

	model.fit(training_queries, validation_queries=validation_queries)

	logging.info('================================================================================')

	test_ranks = model.predict_rankings(test_queries)
	dcg_score = NDCG(test_queries, test_ranks).mean_ndcg()
	map_score = MAP(test_queries, test_ranks).mean_average_precision()
	dcg_folds_scores.append(dcg_score)
	map_folds_scores.append(map_score)

	#evaluate nDCG10 on test queries for best model
	logging.info('%s on the test queries: %.8f'
	             % ('NDCG@10', dcg_score))

	#evaluate MAP on test queries
	logging.info('%s on the test queries: %.8f'
	             % ('MAP', map_score))

	with open('folds_results.txt', 'a') as f:
		f.write("Fold %d \n" %(i))
		f.write("nDCG@10 %f\n" %(dcg_score))
		f.write("MAP %f\n" %(map_score))

	#save the model
	model.save('models/LambdaMART_Fold_' + str(i))
	# filename = 'models/LambdaMART_L7_S0.1_E50_nDCG@10'
	# model = LambdaMART.load(filepath=filename)


#average the ndcgs and the maps
dcg_mean = np.mean(dcg_folds_scores)
#evaluate nDCG10 on test queries for best model
logging.info('Average %s on the test queries from all folds: %.8f'
             % ('NDCG@10', dcg_mean))

map_mean = np.mean(map_folds_scores)
#evaluate MAP on test queries
logging.info('Average %s on the test queries from all folds: %.8f'
             % ('MAP', map_mean))

with open('folds_results.txt', 'a') as f:
	f.write("Average Results \n")
	f.write("nDCG@10 %f\n" %(dcg_mean))
	f.write("MAP %f\n" %(map_mean))
