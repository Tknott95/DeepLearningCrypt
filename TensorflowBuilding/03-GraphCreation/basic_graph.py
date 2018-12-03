import tensorflow as tf

a = tf.constant(5)
b = tf.constant(2)
c = tf.constant(3)

d = tf.multiply(a,b)
e = tf.add(c,b)
f = tf.subtract(d,e)

# Now we have a graph (looks lke a binary tree) for tf with tensors
sess = tf.Session()
outs = sess.run(f)
sess.close()
print("\n outs = {}".format(outs))

