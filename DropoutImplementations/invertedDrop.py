import numpy as np
# Following Code Inside Unreleased OReilly Book Here

network.p = 0.5
def train_step(network, X):
    # Forward Pass For a 3-Layer Neural Net
    Layer1 = np.maximum(0, np.dot(network.W1, X) + network.b1)
    # First Dropout Mask
    Dropout1 = ((np.random.rand(*Layer1.shape) < network.p) / network.p
    Layer1 *= Dropout1 # <- First Drop

    Layer2 = np.maximum(0, np.dot(network.W2, Layer1) + network.b2)
    # second dropout mask, note that we divide by p
    Dropout2 = ((np.random.rand(*Layer2.shape) < network.p) / network.p
    Layer2 *= Dropout2 # <- second drop!
    Output = np.dot(network.W3, Layer2) + network.b3
    # backward pass: compute gradients... (not shown)
    # perform parameter update... (not shown)
def predict(network, X):
    Layer1 = np.maximum(0, np.dot(network.W1, X) + network.b1)
    Layer2 = np.maximum(0, np.dot(network.W2, Layer) + network.b2)
    Output = np.dot(network.W3, Layer2) + network.b3
    return Output
