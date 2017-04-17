from Task_Helper import numpify, normalize_data, import_data
import tensorflow as tf
import pandas as pd



train_data, vali_data, test_data = import_data()

data_dimension = 784
first_layer_size = 256
learning_rate = 0.55

# Graph inputs
x = tf.placeholder(tf.float32, [None, data_dimension])
y = tf.placeholder(tf.float32, [None, data_dimension])

weights = {
    'layer_1': tf.Variable(tf.random_normal([data_dimension, first_layer_size], stddev=0.1)),
    'output': tf.Variable(tf.random_normal([first_layer_size, 1], stddev=0.1))
}
biases = {
    'layer_1': tf.Variable(tf.zeros([first_layer_size])),
    'output': tf.Variable(tf.zeros([1]))
}

x_y = tf.concat([x, y], 1)
x_y_flipped = tf.concat([y, x], 1)

layer_1_original = tf.nn.relu(tf.matmul(x_y, weights['layer_1']) + biases['layer_1'])
layer_1_dual = tf.nn.relu(tf.matmul(x_y_flipped, weights['layer_1']) + biases['layer_1'])

layer_1 = tf.concat([layer_1_original, layer_1_dual], 1)
layer_1_flipped = tf.concat([layer_1_dual, layer_1_original], 1)

intemediate_output_1 = tf.matmul(layer_1, weights['output']) + biases['output']
intemediate_output_2 = tf.matmul(layer_1_flipped, weights['output']) + biases['output']
output_1 = tf.divide(intemediate_output_1, tf.add(intemediate_output_1, intemediate_output_2))
output_2 = tf.divide(intemediate_output_2, tf.add(intemediate_output_1, intemediate_output_2))


# Softmax
cross_entropy_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=y_hat))
train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(cross_entropy_loss)

error = tf.reduce_mean(tf.cast(tf.not_equal(tf.argmax(y_hat, 1), tf.argmax(y, 1)), tf.float32))
get_predictions_and_values = (tf.argmax(y, 1), tf.argmax(y_hat, 1))

training_errors = []
test_errors = []

with tf.Session() as session:
    tf.global_variables_initializer().run()
    
#    if os.path.isfile(checkpoint_folder + "/" + checkpoint_name):
#        load_model(session, checkpoint_folder, checkpoint_name)
    
    else:   
        print("Running...")
        for i in range(5000):
            batch_x, batch_y = mnist_data.train.next_batch(300)
            session.run(train_step, feed_dict={x: batch_x, y: batch_y})
            
            if(np.mod(i, 100) == 0):
                training_errors.append(session.run(error, feed_dict={x: mnist_data.train.images, y: mnist_data.train.labels}))
                test_errors.append(session.run(error, feed_dict={x: mnist_data.test.images, y: mnist_data.test.labels}))
    
        y_true, y_pred = session.run(get_predictions_and_values, feed_dict={x: mnist_data.test.images, y: mnist_data.test.labels})
        #save_model(session, checkpoint_folder, checkpoint_name)

    training_error = session.run(error, feed_dict={x: mnist_data.train.images, y: mnist_data.train.labels})
    test_error = session.run(error, feed_dict={x: mnist_data.test.images, y: mnist_data.test.labels})
    print("Final training error", training_error)
    print("Final test error", test_error)