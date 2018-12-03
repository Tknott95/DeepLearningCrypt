import numpy as _np
import tensorflow as _tf

ex = _tf.constant([
    [1,2,3],
    [4,5,6]
    ])
print('\n is a 2x2 matrix \n List Input: {}'.format(ex.get_shape()))

ex = _tf.constant(_np.array([
    [
        [1,2,3],[4,5,6]
    ],
    [
        [3,1,4],[1,2,4]
    ]
    ]))

print('\n 3D NumPy 2x2x3array input: {}'.format(ex.get_shape()))
