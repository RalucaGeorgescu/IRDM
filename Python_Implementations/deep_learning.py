from Task_Helper import import_data, load_tf_model, save_tf_model
from Pairwise_Helper import get_pairs_as_batch, get_pairs_as_full_dataset
import tensorflow as tf
import numpy as np
import os

checkpoint_folder = 'Saved_Models'
checkpoint_name = 'deep_learning'

train_data, vali_data, test_data = import_data()

train_data = train_data[:1000]
vali_data = vali_data[:1000]
test_data = test_data[:1000]

batch_size = 300
data_dimension = 137
first_layer_size = 544
learning_rate = 0.0001

x_1 = tf.placeholder(tf.float32, [None, data_dimension])
x_2 = tf.placeholder(tf.float32, [None, data_dimension])

relevance = tf.placeholder(tf.int8, [None, 2])
relevance_max = tf.reshape(tf.reduce_max(relevance, axis=1), [-1,1])
relevance_max_bc = tf.concat([relevance_max, relevance_max], axis=1)
relevance_logits = tf.cast(tf.equal(relevance, relevance_max_bc), dtype=tf.float32)

weights = {
    'layer_1': tf.get_variable("w_layer_1", shape=[data_dimension * 2, first_layer_size], initializer=tf.contrib.layers.xavier_initializer()),
    'output': tf.get_variable("w_output", shape=[first_layer_size * 2, 1], initializer=tf.contrib.layers.xavier_initializer())
}
biases = {
    'layer_1': tf.Variable(tf.zeros([first_layer_size])),
    'output': tf.Variable(tf.zeros([1]))
}

x1_x2 = tf.concat([x_1, x_2], 1)
x1_x2_flipped = tf.concat([x_2, x_1], 1)

layer_1_original = tf.nn.relu(tf.matmul(x1_x2, weights['layer_1']) + biases['layer_1'])
layer_1_dual = tf.nn.relu(tf.matmul(x1_x2_flipped, weights['layer_1']) + biases['layer_1'])

layer_1 = tf.concat([layer_1_original, layer_1_dual], 1)
layer_1_flipped = tf.concat([layer_1_dual, layer_1_original], 1)

output_1 = tf.matmul(layer_1, weights['output']) + biases['output']
output_2 = tf.matmul(layer_1_flipped, weights['output']) + biases['output']

output = tf.concat([output_1, output_2], axis=1)
softmax_output = tf.nn.softmax(output)

cross_entropy_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=relevance_logits, logits=output))
train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(cross_entropy_loss)

error = tf.reduce_mean(tf.squared_difference(tf.cast(relevance_logits, dtype=tf.float32), softmax_output))
compare = tf.argmax(output, 1)

training_errors = []
test_errors = []

with tf.Session() as session:
    tf.global_variables_initializer().run()
    
    if os.path.isfile(checkpoint_folder + "/" + checkpoint_name):
        load_tf_model(session, checkpoint_folder, checkpoint_name)
    
    else:   
        print("Running...")
        for i in range(50):
            x_1_batch, x_2_batch, y_batch = get_pairs_as_batch(train_data, 'train', batch_size)
            feed_dict = {x_1: x_1_batch, x_2: x_2_batch, relevance: y_batch}
            
            session.run(train_step, feed_dict=feed_dict)
            
            if(np.mod(i, 1) == 0):
                training_error = session.run(error, feed_dict=feed_dict)
                training_errors.append(training_error)
                print("Batch training error:", training_error)
                
    
        save_tf_model(session, checkpoint_folder, checkpoint_name)

    x_1_train_full, x_2_train_full, y_train_full = get_pairs_as_full_dataset(train_data, 'train')
    x_1_test_full, x_2_test_full, y_test_full = get_pairs_as_full_dataset(test_data, 'test')
    feed_dict_train = {x_1: x_1_train_full, x_2: x_2_train_full, relevance: y_train_full}
    feed_dict_test = {x_1: x_1_test_full, x_2: x_2_test_full, relevance: y_test_full}
    
    training_error = session.run(error, feed_dict=feed_dict_train)
    test_error = session.run(error, feed_dict=feed_dict_test)
    print("Final training error", training_error)
    print("Final test error", test_error)
