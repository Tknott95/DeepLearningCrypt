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
path_to_file = tf.keras.utils.get_file('shakespeare.txt', 'http://www.gutenberg.org/cache/epub/1497/pg1497.txt') 
# FRANCIS BACON: 'http://www.gutenberg.org/files/56463/56463-0.txt'
# SHAKESPEAR: 'https://storage.googleapis.com/download.tensorflow.org/data/shakespeare.txt')
# MARCUS AURELIOUS: http://www.gutenberg.org/cache/epub/15877/pg15877.txt
# Reading Bacons work as "spear shaker" - Athena did this at ignorance.
# PLATO: http://www.gutenberg.org/cache/epub/1497/pg1497.txt
# HOMER LLIAD: http://www.gutenberg.org/cache/epub/6130/pg6130.txt
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

optimizer = tf.train.AdamOptimizer()

# using sparse_softmax_cross_entropy so that we don't have to create one-hot vectors
def loss_function(real, preds):
    return tf.losses.sparse_softmax_cross_entropy(labels=real, logits=preds)

checkpoint_dir = './training_checkpoints'
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
checkpoint = tf.train.Checkpoint(optimizer=optimizer,
                                 model=model)

                                 # Training step

EPOCHS = 20

for epoch in range(EPOCHS):
    start = time.time()
    
    # initializing the hidden state at the start of every epoch
    hidden = model.reset_states()
    
    for (batch, (inp, target)) in enumerate(dataset):
          with tf.GradientTape() as tape:
              # feeding the hidden state back into the model
              # This is the interesting step
              predictions, hidden = model(inp, hidden)
              
              # reshaping the target because that's how the 
              # loss function expects it
              target = tf.reshape(target, (-1,))
              loss = loss_function(target, predictions)
              
          grads = tape.gradient(loss, model.variables)
          optimizer.apply_gradients(zip(grads, model.variables))

          if batch % 100 == 0:
              print ('Epoch {} Batch {} Loss {:.4f}'.format(epoch+1,
                                                            batch,
                                                            loss))
    # saving (checkpoint) the model every 5 epochs
    if (epoch + 1) % 5 == 0:
      checkpoint.save(file_prefix = checkpoint_prefix)

    print ('Epoch {} Loss {:.4f}'.format(epoch+1, loss))
    print('Time taken for 1 epoch {} sec\n'.format(time.time() - start))

    # run the katest checkpoint brother chucker
    checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))

    # The oracle now forseeeeeeeeeeeyyyyUUhzZZ
    # Evaluation step(generating text using the model learned)

# number of characters to generate
num_generate = 1000

# You can change the start string to experiment
start_string = 'Q'
# converting our start string to numbers(vectorizing!) 
input_eval = [char2idx[s] for s in start_string]
input_eval = tf.expand_dims(input_eval, 0)

# empty string to store our results
text_generated = ''

# hidden state shape == (batch_size, number of rnn units); here batch size == 1
hidden = [tf.zeros((1, units))]
for i in range(num_generate):
    predictions, hidden = model(input_eval, hidden)

    # using argmax to predict the word returned by the model
    predicted_id = tf.argmax(predictions[-1]).numpy()
    
    # We pass the predicted word as the next input to the model
    # along with the previous hidden state
    input_eval = tf.expand_dims([predicted_id], 0)
    
    text_generated += idx2char[predicted_id]

print (start_string + text_generated)