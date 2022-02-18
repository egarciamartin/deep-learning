import numpy as np


class LeakyReLU():
    def __init__(self) -> None:
        self.name = 'leaky_relu'

    def __call__(self, z):
        """
        if x>0: x else: 0.01x
        np.where: if z>0, then z, otherwise z*alpha 
        """
        alpha = 0.01
        return np.where(z > 0, z, z*alpha)

    def backward(self, a, z):
        alpha = 0.01
        return np.where(z > 0, 1, z*alpha)


class Softmax():
    def __init__(self) -> None:
        self.name = 'softmax'

    def __call__(self, z):
        # e_x = np.exp(z - np.max(z))
        e_x = np.exp(z)
        return e_x / np.sum(e_x, axis=0)

    def backward(self, y, y_hat):
        """

        Computed atm with the cross entropy
        """
        return (y_hat - y).T


class ReLU():
    def __init__(self) -> None:
        self.name = 'relu'

    def __call__(self, z):
        return z * (z > 0)

    def backward(self, a, z):
        return 1 * (z > 0)


class Linear():
    def __init__(self) -> None:
        self.name = 'linear'

    def __call__(self, z):
        return z

    def backward(self, a, z):
        """
        Returning the derivative
        """
        return 1


class Tanh():
    def __init__(self) -> None:
        self.name = 'tanh'

    def __call__(self, z):
        return np.tanh(z)

    def backward(self, a, z):
        """
        1 - fx^2
        """
        return 1 - (a ** 2)


class Sigmoid():
    def __init__(self) -> None:
        self.name = 'sigmoid'

    def __call__(self, z):
        return 1 / (1 + np.exp(-z))

    def backward(self, a, z):
        """
        f(x)(1-f(x))
        """
        return a * (1 - a)
