import numpy as np

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
