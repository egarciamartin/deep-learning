import numpy as np


def leaky_relu(z):
    """
    if x>0: x else: 0.01x
    np.where: if z>0, then z, otherwise z*alpha 
    """
    alpha = 0.01
    return np.where(z > 0, z, z*alpha)


def relu(z):
    return z * (z > 0)


def tanh(z):
    return np.tanh(z)


def sigmoid(z):
    return 1 / (1 + np.exp(-z))
