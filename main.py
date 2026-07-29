import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

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


def col_operation(A, W, B):
    X = (W @ A) + B

    Y = 1 / (1 + np.exp(-X))

    return Y

def forward_prop(X, W1, B1, W2, B2, W3, B3):
    A1 = col_operation(X, W1, B1)
    A2 = col_operation(A1, W2, B2)
    A3 = col_operation(A2, W3, B3)

    return A3

def run_infr(X, W1, B1, W2, B2, W3, B3):
    Y = forward_prop(X, W1, B1, W2, B2, W3, B3)

    return Y.index(max(Y))











