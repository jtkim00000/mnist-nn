import numpy as numpy
import matplotlib import pyplot as pyplot
import pasdas as pd

# data = //some data
# using some external database for MNIST

data = np.array(data)
m, n = data.shape()
np.random.shuffle(data)


data_dev = np.transpose(data[0:1000])

Y_dev = data_dev[0]
X_dev = data_dev[1:n]

data_train = np.transpose(data[1000:m])
Y_train = data_train[0]
X_train = data_train[1:n]

def init_params():

    W1 = np.random.rand(128, 784)
    B1 = np.random.rand(128, 1)

    W2 = np.random.rand(64, 128)
    B2 = np.random.rand(64, 1)

    W3 = np.random.rand(10, 64)
    B3 = np.random.rand(10, 1)

    return W1, B1, W2 B2, W3, B3

def ReLU(Z):
    return np.maximum(0, Z) 

def softmax(Z):
    return np.exp(Z) / np.sum(np.exp(Z))

def forward_prop(W1, B1, W2, B2, W3, B3, X):

    Z1 = W1 @ X + B1
    A1 = ReLU(Z1)

    Z2 = W2 @ A1 + B2
    A2 = ReLU(Z2)

    Z3 = W3 @ A2 + B3
    A3 = softmax(Z3)

    return A3


def one_hot(Y):
    one_hot_Y = np.zeros((Y.size, Y.max() + 1))
    one_hot_Y[np.arrange(y.size), Y] = 1
    return one_hot_Y = np.transpose(one_hot_Y)

def deriv_ReLU(Z):
    return Z > 0

def back_prop(Z1, A1, Z2, A2, Z3, A3, Y):
    one_hot_Y = one_hot(Y)

    m = Y.size

    dZ2 = A2 - one_hot_Y

    dW2 = 1 / m * dZ2 @ np.transpose(A1)

    dB2 = 1 / m * sp.sum(dZ2)

    dZ1 = (np.transpose(W2)).dot(dZ2) * deriv_ReLU(Z1)

    dW1 = 1 / m * dZ2.dot(np.transpose(A1))
    dB1 = 1/ m * np.sum(dZ2, 2)

    return dW1, dB1, dW2, dB2


def update_params(W1, B1, W2, B2, dW1, dW2, dB2, alpha):
    












