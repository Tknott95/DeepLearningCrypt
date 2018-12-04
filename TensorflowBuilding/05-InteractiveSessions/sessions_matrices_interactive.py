import tensorflow as _tf
import numpy as _np

sess = _tf.InteractiveSession()
ex = _tf.linspace(0.0, 4.0, 5)
print('The content of "ex":\n {}\n'.format(ex.eval()))
sess.close()

print("\n____\n")
twoThree = _tf.constant([
    [3,1,4],
    [4,2,8]
])

print('List Input {}'.format(twoThree.get_shape()))
twoTwoThree = _tf.constant(_np.array([
    [[3,1,4],
    [4,2,8]],
    [[3,0,3],
    [4,8,4]]
]))

print('3D Numpy array input: {}'.format(twoTwoThree.get_shape()))

sess2 = _tf.InteractiveSession()
ex44 = _tf.linspace(0.0, 4.0, 7)
print('The content of "c":\n {}\n'.format(ex44.eval()))
sess2.close()
