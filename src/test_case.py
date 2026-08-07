import numpy as np
from sklearn.datasets import fetch_openml

mnist = fetch_openml(
    "mnist_784",
    version=1,
    as_frame=False
)

data = mnist.data

data = np.array(data)

X = mnist.data.astype(np.float32) / 255.0

test_case = X[0:1023]

print(test_case[50])

np.savez(
    "test_case.npz",
    test_case=test_case,
)