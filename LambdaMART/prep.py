import numpy as np
import logging
import rankpy
from rankpy.queries import Queries

# Turn on logging.
logging.basicConfig(format='%(asctime)s : %(message)s', level=logging.INFO)

for i in xrange(1, 6):	
	# Load the query datasets.
	training_queries = Queries.load_from_text('MSLR-WEB10K/Fold' + str(i) + '/train.txt')
	validation_queries = Queries.load_from_text('MSLR-WEB10K/Fold' + str(i) + '/vali.txt')
	test_queries = Queries.load_from_text('MSLR-WEB10K/Fold' + str(i) + '/test.txt')

	logging.info('================================================================================')

	# Save them to binary format ...
	training_queries.save('MSLR-WEB10K/Fold' + str(i) + '/training')
	validation_queries.save('MSLR-WEB10K/Fold' + str(i) + '/validation')
	test_queries.save('MSLR-WEB10K/Fold' + str(i) + '/test')

	# ... because loading them will be then faster.
	training_queries = Queries.load('MSLR-WEB10K/Fold' + str(i) + '/training')
	validation_queries = Queries.load('MSLR-WEB10K/Fold' + str(i) + '/validation')
	test_queries = Queries.load('MSLR-WEB10K/Fold' + str(i) + '/test')

	logging.info('================================================================================')
