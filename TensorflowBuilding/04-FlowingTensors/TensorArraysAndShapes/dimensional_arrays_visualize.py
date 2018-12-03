import numpy as _np
import tensorflow as _tf

ex = _tf.constant([
    [1,2,3],
    [4,5,6]
    ])
print('\n is a 2x2 matrix \n List Input: {}'.format(ex.get_shape()))
