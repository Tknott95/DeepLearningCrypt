# Import TensorFlow >= 1.10 and enable eager execution
import tensorflow as tf
from model import Model
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

# unique contains all the unique characters in the file
unique = sorted(set(text))

# creating a mapping from unique characters to indices
char2idx = {u:i for i, u in enumerate(unique)}
idx2char = {i:u for i, u in enumerate(unique)}

# setting the maximum length sentence we want for a single input in characters
max_length = 100

# length of the vocabulary in chars
vocab_size = len(unique)

# the embedding dimension 
embedding_dim = 256

# number of RNN (here GRU) units
units = 1024

# batch size 
BATCH_SIZE = 64

# buffer size to shuffle our dataset
BUFFER_SIZE = 10000

input_text = []
target_text = []

for f in range(0, len(text)-max_length, max_length):
    inps = text[f:f+max_length]
    targ = text[f+1:f+1+max_length]

    input_text.append([char2idx[i] for i in inps])
    target_text.append([char2idx[t] for t in targ])
    
print (np.array(input_text).shape)
print (np.array(target_text).shape)

dataset = tf.data.Dataset.from_tensor_slices((input_text, target_text)).shuffle(BUFFER_SIZE)
dataset = dataset.batch(BATCH_SIZE, drop_remainder=True)

model = Model(vocab_size, embedding_dim, units, BATCH_SIZE)
