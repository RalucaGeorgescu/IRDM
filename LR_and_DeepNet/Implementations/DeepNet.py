from Task_Helper import import_data, load_tf_model, save_tf_model
from Pairwise_Helper import get_pairs_as_batch, get_pairs_as_full_dataset
import tensorflow as tf
import numpy as np

tf.reset_default_graph()

checkpoint_folder = 'Saved_Models'
checkpoint_name = 'deep_learning'

train_data, vali_data, test_data = import_data()

batch_size = 300
data_dimension = 137
first_layer_size = 720
second_layer_size = 360
third_layer_size = 180

learning_rate = 1e-2
dropout_rate = 0.1

x_1 = tf.placeholder(tf.float32, [None, data_dimension])
x_2 = tf.placeholder(tf.float32, [None, data_dimension])

relevance = tf.placeholder(tf.int8, [None, 2])
relevance_max = tf.reshape(tf.reduce_max(relevance, axis=1), [-1, 1])
relevance_max_bc = tf.concat([relevance_max, relevance_max], axis=1)
relevance_logits = tf.cast(tf.equal(relevance, relevance_max_bc), dtype=tf.float32)

weights = {
    'layer_1': tf.get_variable("w_layer_1", shape=[data_dimension * 2, first_layer_size], initializer=tf.contrib.layers.xavier_initializer()),
    'layer_2': tf.get_variable("w_layer_2", shape=[first_layer_size * 2, second_layer_size], initializer=tf.contrib.layers.xavier_initializer()),
    'layer_3': tf.get_variable("w_layer_3", shape=[second_layer_size * 2, third_layer_size], initializer=tf.contrib.layers.xavier_initializer()),
    'output': tf.get_variable("w_output", shape=[first_layer_size * 2, 1], initializer=tf.contrib.layers.xavier_initializer())
}
biases = {
    'layer_1': tf.Variable(tf.zeros([first_layer_size])),
    'layer_2': tf.Variable(tf.zeros([second_layer_size])),
    'layer_3': tf.Variable(tf.zeros([third_layer_size])),
    'output': tf.Variable(tf.zeros([1]))
}

x_1_drop = tf.nn.dropout(x_1, dropout_rate)
x_2_drop = tf.nn.dropout(x_2, dropout_rate)

x1_x2 = tf.concat([x_1_drop, x_2_drop], 1)
x1_x2_flipped = tf.concat([x_2_drop, x_1_drop], 1)

layer_1_original = tf.nn.dropout(tf.nn.relu(tf.matmul(x1_x2, weights['layer_1']) + biases['layer_1']), dropout_rate)
layer_1_dual = tf.nn.dropout(tf.nn.relu(tf.matmul(x1_x2_flipped, weights['layer_1']) + biases['layer_1']), dropout_rate)

layer_1 = tf.concat([layer_1_original, layer_1_dual], 1)
layer_1_flipped = tf.concat([layer_1_dual, layer_1_original], 1)

layer_2_original = tf.nn.dropout(tf.nn.relu(tf.matmul(layer_1, weights['layer_2']) + biases['layer_2']), dropout_rate)
layer_2_dual = tf.nn.dropout(tf.nn.relu(tf.matmul(layer_1_flipped, weights['layer_2']) + biases['layer_2']), dropout_rate)

layer_2 = tf.concat([layer_2_original, layer_2_dual], 1)
layer_2_flipped = tf.concat([layer_2_dual, layer_2_original], 1)

layer_3_original = tf.nn.dropout(tf.nn.relu(tf.matmul(layer_2, weights['layer_3']) + biases['layer_3']), dropout_rate)
layer_3_dual = tf.nn.dropout(tf.nn.relu(tf.matmul(layer_2_flipped, weights['layer_3']) + biases['layer_3']), dropout_rate)

layer_3 = tf.concat([layer_3_original, layer_3_dual], 1)
layer_3_flipped = tf.concat([layer_3_dual, layer_3_original], 1)

output_1 = tf.matmul(layer_1, weights['output']) + biases['output']
output_2 = tf.matmul(layer_1_flipped, weights['output']) + biases['output']

output = tf.concat([output_1, output_2], axis=1)
sigmoid_output = tf.nn.sigmoid(output)

error = tf.reduce_mean(tf.squared_difference(tf.cast(relevance_logits, dtype=tf.float32), sigmoid_output))
train_step = tf.train.RMSPropOptimizer(learning_rate).minimize(error)

compare = tf.greater(output_1, output_2)

training_errors = []

def train_model(train_data, test_data):
    with tf.Session() as session:
        tf.global_variables_initializer().run()
        
        print("Running...")
        for i in range(20000):
            x_1_batch, x_2_batch, y_batch = get_pairs_as_batch(train_data, 'train', batch_size)
            feed_dict = {x_1: x_1_batch, x_2: x_2_batch, relevance: y_batch}
            
            session.run(train_step, feed_dict=feed_dict)
            
            if(np.mod(i, 1000) == 0):
                training_error = session.run(error, feed_dict=feed_dict)
                training_errors.append(training_error)
                print("Iteration: {} Batch training error: {}".format(i, training_error))
                
        save_tf_model(session, checkpoint_folder, checkpoint_name)
    
        x_1_train_full, x_2_train_full, y_train_full = get_pairs_as_full_dataset(train_data, 'train')
        x_1_test_full, x_2_test_full, y_test_full = get_pairs_as_full_dataset(test_data, 'test')
        feed_dict_train = {x_1: x_1_train_full, x_2: x_2_train_full, relevance: y_train_full}
        feed_dict_test = {x_1: x_1_test_full, x_2: x_2_test_full, relevance: y_test_full}
        
        training_error = session.run(error, feed_dict=feed_dict_train)
        test_error = session.run(error, feed_dict=feed_dict_test)
        
        print("Final training error", training_error)
        print("Final test error", test_error)

def compare_with_NN(session, document_1, document_2):
    document_1 = np.array([document_1])
    document_2 = np.array([document_2])
    
    feed_dict = {x_1: document_1, x_2: document_2}
    return session.run(compare, feed_dict=feed_dict)
    
def cmpNN_sort(list_of_documents):
    indices = np.arange(min(list_of_documents.index.values), max(list_of_documents.index.values) + 1)
    list_of_documents = list_of_documents.as_matrix()
    
    # Sort by descending order
    with tf.Session() as session:
        tf.global_variables_initializer().run()
        load_tf_model(session, checkpoint_folder, checkpoint_name)
        
        for pass_doc in range(list_of_documents.shape[0] - 1, 0, -1):
            for i in range(pass_doc):                    
                if compare_with_NN(session, list_of_documents[i], list_of_documents[i + 1]):
                    temp = list_of_documents[i]
                    list_of_documents[i] = list_of_documents[i + 1]
                    list_of_documents[i + 1] = temp
                    
                    temp_indices = indices[i]                     
                    indices[i] = indices[i+1]
                    indices[i+1] = temp_indices
        return list_of_documents, indices

def relevance_sort(list_of_documents):
    return list_of_documents.sort_values(by='relevance', ascending=False)

def precision_at_r(self, rel_scores, rank):
        rel_scores = np.asarray(rel_scores)[:rank]
        return np.mean(rel_scores)

def average_precision(self, binary_relevances):
    num_relevant = np.sum(binary_relevances)
    if num_relevant == 0:
        denom = 0
    else:
        denom = 1. / num_relevant
    avg_precision =  sum([(sum(binary_relevances[:ind+1]) / (ind + 1.)) * denom for ind, val in enumerate(binary_relevances) if val])
    return avg_precision

def test_sort_for_MAP_and_NDCG():
    
    def precision_at_r(rel_scores, rank):
        rel_scores = np.asarray(rel_scores)[:rank]
        return np.mean(rel_scores)

    def average_precision(binary_relevances):
        num_relevant = np.sum(binary_relevances)
        if num_relevant == 0:
            denom = 0
        else:
            denom = 1. / num_relevant
        avg_precision =  sum([(sum(binary_relevances[:ind+1]) / (ind + 1.)) * denom for ind, val in enumerate(binary_relevances) if val])
        return avg_precision
    
    def dcg(rel_scores, rank=10):
        rel_scores = np.asarray(rel_scores)[:rank]
        num_scores = len(rel_scores)
        if num_scores == 0:
            return 0
        gains = 2**rel_scores -1
        discounts = np.log2(np.arange(num_scores)+2)
        return np.sum(gains/discounts)

    def ndcg(rank_rel_scores, rank=10):
        dcg_score = dcg(rank_rel_scores, rank)
        opt_dcg_score = dcg(sorted(rank_rel_scores, reverse=True), rank)
        if opt_dcg_score == 0:
            return 0
        return dcg_score / opt_dcg_score
        
    all_ndcgs = []    
    grouped_data = test_data.groupby('qid')
    for _, group in grouped_data:
        true_relevances = group.pop('relevance')
        
        _, cmpNN_sorted_indices = cmpNN_sort(group)
        cmpNN_sorted_indices -= max(cmpNN_sorted_indices)
        
        ranked_relevances = true_relevances.iloc[cmpNN_sorted_indices]
        all_ndcgs.append(ndcg(ranked_relevances, 10))
    
    all_precisions = []
    grouped_data = test_data.groupby('qid')
    for _, group in grouped_data:
        true_relevances = group.pop('relevance')
        
        _, cmpNN_sorted_indices = cmpNN_sort(group)
        cmpNN_sorted_indices -= max(cmpNN_sorted_indices)
        
        ranked_relevances = true_relevances.iloc[cmpNN_sorted_indices]
        binary_relevances = (ranked_relevances > 0).astype(int)
        all_precisions.append(average_precision(binary_relevances))
    
    return np.mean(all_ndcgs), np.mean(all_precisions)

mean_ndcg, mean_map = test_sort_for_MAP_and_NDCG()
print('Mean NDCG: {} Mean MAP: {}'.format(mean_ndcg, mean_map))
