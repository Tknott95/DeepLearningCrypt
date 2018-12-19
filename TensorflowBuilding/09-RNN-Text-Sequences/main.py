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


