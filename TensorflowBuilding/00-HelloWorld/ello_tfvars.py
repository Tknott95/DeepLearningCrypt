import tensorflow as tf

h = tf.constant('!Hola')
w = tf.constant('AmigBrooos')
hw = h + w
# This way is so we may check the Tensor90 whenever
# Calling Vars normal returns: !Hola AmigBroos# Calling TF way returns: Tensor("add:0", shape=(), dtype=string)
print(hw)

ans = sess.run(hw)

