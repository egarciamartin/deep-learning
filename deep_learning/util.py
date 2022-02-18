import numpy as np


def one_hot(y):
    n_classes = np.max(y) + 1
    return np.eye(n_classes)[y]
