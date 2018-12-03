import tensorflow as _tf

sess = _tf.InteractiveSession()
ex = _tf.linspace(0.0, 4.0, 5)
print('The content of "ex":\n {}\n'.format(ex.eval()))
sess.close()

print("\n____\n")
twoThree = _tf.constant([
    [3,1,4],
    [4,2,8]
])
