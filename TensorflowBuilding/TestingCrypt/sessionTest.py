import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
d33p = tf.constant('d33p')
sess = tf.Session()
sess.run(d33p)
