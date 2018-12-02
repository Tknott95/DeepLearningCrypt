import tensorflow as tf

h = tf.constant('Suhhh')
w = tf.constant('waynes world')
hw = h + w

with tf.Session() as sess:
    exmple = sess.run(hw)

print(exmple)
