import numpy as np


W1 = np.random.rand(16, 784) * 0.005
B1 = np.random.rand(16,  1)

W2 = np.random.rand(16, 16) * 0.005
B2 = np.random.rand(16, 1)

W3 = np.random.rand(10, 16) * 0.005
B3 = np.random.rand(10, 1)

np.savez(
    "mnist_network.npz",
    W1=W1,
    B1=B1,
    W2=W2,
    B2=B2,
    W3=W3,
    B3=B3
)