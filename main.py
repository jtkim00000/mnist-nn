import numpy as np
import matplotlib.pyplot as plt

import forward_propagation
import nn_visualizer

# Initializing Shapes

# Input Vector
X = np.zeros(784)

# Activations
A1 = np.zeros(64)
A2 = np.zeros(16)
A3 = np.zeros(10)

# Weights
W1 = np.zeros(64, 784)
W2 = np.zeros(16, 64)
W3 = np.zeros(10, 16)

# Biases
B1 = np.zeros(64)
B2 = np.zeros(16)
B3 = np.zeros(10)

def randomize():
    X = np.random.rand()







