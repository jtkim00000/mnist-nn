import numpy as np
from sklearn.datasets import fetch_openml

# ==================================================
#                  INITIALIZATION
# ==================================================

mnist = fetch_openml(
    "mnist_784",
    version=1,
    as_frame=False
)

data = mnist.data

data = np.array(data)

X = mnist.data.astype(np.float32) / 255.0
Y = mnist.target.astype(np.int32)

# Split train/test
X_train = X[:60000]
Y_train = Y[:60000]

X_test = X[60000:]
Y_test = Y[60000:]

accuracy = 0.0

W1 = np.random.rand(16, 784) * 0.005
B1 = np.random.rand(16,  1)

W2 = np.random.rand(16, 16) * 0.005
B2 = np.random.rand(16, 1)

W3 = np.random.rand(10, 16) * 0.005
B3 = np.random.rand(10, 1)

# ==================================================
#                  INITIAL FUNCTIONS
# ==================================================

def col_operation(A, W, B):
    X = (W @ A) + B

    Y = 1 / (1 + np.exp(-X))

    return Y

def forward_prop(X, W1, B1, W2, B2, W3, B3):
    A1 = col_operation(X, W1, B1)
    A2 = col_operation(A1, W2, B2)
    A3 = col_operation(A2, W3, B3)

    return A1, A2, A3

def make_target(label):

    target = np.zeros((10,1))
    target[label] = 1

    return target

def calculate_accuracy(X_data, Y_data, W1, B1, W2, B2, W3, B3):

    correct = 0

    for i in range(len(X_data)):

        X = X_data[i].reshape(784,1)

        output, _, _ = forward_prop(
            X,
            W1,B1,
            W2,B2,
            W3,B3
        )

        prediction = np.argmax(output)

        if prediction == Y_data[i]:
            correct += 1

    return correct / len(X_data)

def back_prop():

# ==================================================
#                  TRAINING LOOP
# ==================================================

while accuracy < 0.95:

    epoch += 1

    print("Starting epoch:", epoch)


    for i in range(len(X_train)):

        X = X_train[i].reshape(784,1)
        target = make_target(Y_train[i])

        A1, A2, A3 = forward_prop(
            X,
            W1,B1,
            W2,B2,
            W3,B3
        )

        dW1,dB1,dW2,dB2,dW3,dB3 = backward_prop(
            X,
            target,
            A1,
            A2,
            A3,
            W1,W2,W3
        )


        # Update
        W1 -= learning_rate*dW1
        B1 -= learning_rate*dB1

        W2 -= learning_rate*dW2
        B2 -= learning_rate*dB2

        W3 -= learning_rate*dW3
        B3 -= learning_rate*dB3


    accuracy = calculate_accuracy(
        X_test,
        Y_test,
        W1,B1,
        W2,B2,
        W3,B3
    )


    print("Accuracy:", accuracy)


print("Training complete!")

# ==================================================
#                  SAVING PARAMETERS
# ==================================================

if(accuracy >= 0.95):
    np.savez(
        "mnist_network.npz",
        W1=W1,
        B1=B1,
        W2=W2,
        B2=B2,
        W3=W3,
        B3=B3
    )