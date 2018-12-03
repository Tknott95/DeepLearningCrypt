import tensorflow as tf
print(tf.get_default_graph())

g = tf.Graph()
print(g)


print("\n")
g2 = tf.Graph()
a = tf.constant(5)

print(a.graph is g2)
print(a.graph is tf.get_default_graph())

print("\n")
g3 = tf.get_default_graph()
g4 = tf.Graph() # reused to mentally see for these basic examples #sloppyForLearning

print(g3 is tf.get_default_graph())

with g4.as_default():
    print(g3 is tf.get_default_graph())

print(g3 is tf.get_default_graph())

print("\n____FETCHES____\n")


z = tf.constant(4)
y = tf.constant(17)
x = tf.constant(33)

v = tf.multiply(z,x)
m = tf.add(y,z)
n = tf.subtract(x,y)

with tf.Session() as sess:
    fetches = [z,y,x,v,m,n]
    outs = sess.run(fetches)

print("outs = {}".format(outs))
print(type(outs[0]))

outlier_x = tf.constant(3.14)
print(outlier_x)
