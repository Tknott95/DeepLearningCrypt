import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

_mnist = input_data.read_data_sets("/tmp/data", one_hot=True)

ele_size = 28
time_steps = 28
number_classes = 10
batch_size = 128
hidden_layer_size = 128

# for Tensorboard summs bruhz
LOG_DIR = "logs/RNN_with_summaries"

_inputs = tf.placeholder(tf.float32, shape=[None, time_steps, ele_size], name='inputs')
y = tf.placeholder(tf.float32, shape=[None, number_classes], name='labels')

batch_x, batch_y = _mnist.train.next_batch(batch_size)
# Reshape data to get 28 sequences of 28 pixels
batch_x = batch_x.reshape((batch_size, time_steps, ele_size))
# This helper function, taken from the official TensorFlow documentation, # simply adds some ops that take care of logging summaries
def variable_summaries(var):
    with tf.name_scope('summaries'):
          mean = tf.reduce_mean(var)
          tf.summary.scalar('mean', mean)
    with tf.name_scope('stddev'):
        stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
        tf.summary.scalar('stddev', stddev)
        tf.summary.scalar('max', tf.reduce_max(var))
        tf.summary.scalar('min', tf.reduce_min(var))
        tf.summary.histogram('histogram', var)
#         with tf.name_scope('rnn_weights'):
#             with tf.name_scope("W_x"):
#                 Wx = tf.Variable(tf.zeros([ele_size, hidden_layer_size])) variable_summaries(Wx) 
#             with tf.name_scope("W_h"):
#                 Wh = tf.Variable(tf.zeros([hidden_layer_size, hidden_layer_size])) variable_summaries(Wh)
#             with tf.name_scope("Bias"):
#                 b_rnn = tf.Variable(tf.zeros([hidden_layer_size])) variable_summaries(b_rnn)
# def rnn_step(previous_hidden_state,x):
#             current_hidden_state = tf.tanh(
#                 tf.matmul(previous_hidden_state, Wh) +
#                 tf.matmul(x, Wx) + b_rnn)
# return current_hidden_state
# # Processing inputs to work with scan function
# # Current input shape: (batch_size, time_steps, element_size) processed_input = tf.transpose(_inputs, perm=[1, 0, 2])
# # Current input shape now: (time_steps, batch_size, element_size)
# initial_hidden = tf.zeros([batch_size,hidden_layer_size]) # Getting all state vectors across time all_hidden_states = tf.scan(rnn_step,
#                                 processed_input,
#                                 initializer=initial_hidden,
#                                 name='states')
