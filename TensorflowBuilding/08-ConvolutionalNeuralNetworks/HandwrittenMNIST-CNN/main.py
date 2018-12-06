from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as _tf
import numpy as _np

from network_layers import convolutional_layer, max_pool_2x2, full_layer

# using 'Learning Tensorflow O'Reilly book to code this
DATA_DIR = '/tmp/data'
MINIBATCH_SIZE = 50
STEPS = 5000

mnist = input_data.read_data_sets(DATA_DIR, one_hot=True)

x = _tf.placeholder(_tf.float32, shape=[None, 784])
y_ = _tf.placeholder(_tf.float32, shape=[None, 10])
x_image = _tf.reshape(x, [-1, 28, 28, 1])

conv1 = convolutional_layer(x_image, _shape=[5,5,1,32])
conv1_pool = max_pool_2x2(conv1)

conv2 = convolutional_layer(conv1_pool, _shape=[5,5,32,64])
conv2_pool = max_pool_2x2(conv2)

conv2_flat = _tf.reshape(conv2_pool, [-1, 7*7*64])
full_1 = _tf.nn.relu(full_layer(conv2_flat, 1024))

keep_prob = _tf.placeholder(_tf.float32)
fulll_drop = _tf.nn.dropout(full_1, keep_prob=keep_prob)

y_conv = full_layer(fulll_drop, 10)

cross_entropy = _tf.reduce_mean(_tf.nn.softmax_cross_entropy_with_logits(logits=y_conv, labels=y_))
train_step = _tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
correct_prediction = _tf.equal(_tf.argmax(y_conv, 1), _tf.argmax(y_, 1))

accuracy = _tf.reduce_mean(_tf.cast(correct_prediction, _tf.float32))

with _tf.Session() as sess:
    sess.run(_tf.global_variables_initializer())
    
    for i in range(STEPS):
        batch = mnist.train.next_batch(MINIBATCH_SIZE)

        if i % 100 == 0:
            train_accuracy = sess.run(accuracy, feed_dict={x: batch[0], y_: batch[1], keep_prob: 1.0})
            print("step {}, training accuracy {}".format(i, train_accuracy))

        sess.run(train_step, feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})

    X = mnist.test.images.reshape(10, 1000, 784)
    Y = mnist.test.labels.reshape(10, 1000, 10)
    test_accuracy = _np.mean([sess.run(accuracy, feed_dict={x: X[i], y_: Y[i], keep_prob: 1.0}) for i in range(10)])

print("test_accuracy: {}".format(test_accuracy))
