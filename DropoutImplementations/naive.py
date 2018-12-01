import numpy as np 
# Following Code Inside Unreleased OReilly Book Here

network.p = 0.5

def train_step(network, X):
    # Forward Pass For a 3-Layer Neural Net
    Layer1 = np.maximum(0, np.dot(network.W1, X) + network.b1)
    # First Dropout Mask
    Dropout1 = (np.random.rand(*Layer1.shape) < network.p)
    Layer1 *= Dropout1 # <- First Drop

    Layer2 = np.maximum(0, np.dot(network.W2, Layer1) + network.b2)
    # 2nd dropout mask
    Dropout2 = (np,random.rand(*Layer2.shape) < network.p)
    # 2nd drop
    Layer2 *= Dropout2

    Output = np.dot(network.W3, Layer2) + network.b3

def predict(network, X):
    # NOTE: we scale the activations
    Layer1 = np.maximum(0, np.dot(network.W1, X) + network.b1) * network.p
    Layer2 = np.maximum(0, np.dot(network.W2, Layer) + network.b2) * network.p
    Output = np.dot(network.W3, Layer2) + network.b3
    return Output

