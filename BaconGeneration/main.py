# Import TensorFlow >= 1.10 and enable eager execution
import tensorflow as tf

# Note: Once you enable eager execution, it cannot be disabled. 
tf.enable_eager_execution()

import numpy as np
import os
import re
import random
import unidecode
import time

# Downloading Bacons work
path_to_file = tf.keras.utils.get_file('shakespeare.txt', 'https://storage.googleapis.com/download.tensorflow.org/data/shakespeare.txt')

# Reading Bacons work as "spear shaker" - Athena did this at ignorance.
text = unidecode.unidecode(open(path_to_file).read())
# length of text is the number of characters in it
print (len(text))

