import tensorflow as _tf

def weight_var(_shape):
    initial = _tf.truncated_normal(_shape, stddev=0.1)
    return _tf.Variable(initial)

def bias_var(_shape):
    initial = _tf.constant(0.1, shape=_shape)
    return _tf.Variable(initial)

def convolutional_2D(_x, _W):
    return _tf.nn.conv2d(_x, _W, strides=[1,1,1,1], padding='SAME')

def max_pool_2x2(_x):
    return _tf.nn.max_pool(_x, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')

def convolutional_layer(_input, _shape):
    W = weight_var(_shape)
    b = bias_var([shape[3]])
    return _tf.nn.relu(convolutional_2D(_input, W) + b)

def full_layer(_input, _size):
    in_size = int(_input.get_shape()[1])
    W = weight_var([in_size, _size])
    b = bias_var([_size])
    return _tf.matmul(_input, W) + b
