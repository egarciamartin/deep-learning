import numpy as np


class Loss:
    def __init__(self) -> None:
        pass


class MAE(Loss):
    """
    Regression
    """

    def __init__(self) -> None:
        super().__init__()

    def loss(self, y, y_hat):
        """
        Returns the MAE between y and y_hat
        """
        return np.sum(np.abs(y_hat - y))

    def backward(self):
        pass


class CrossEntropy(Loss):
    """
    Multi class cross entropy loss
    Expecting the Y vector as one-hot encoded
    Works also for binary classification. 

    Binary: last layer sigmoid activation
    Multi-class: last layer is a softmax
    Regerssion: last layer is just linear, meaning no activation
    """

    def loss(self, y, y_hat):
        m = y.shape[0]
        eps = 1e-15
        y_hat = np.clip(y_hat, eps, 1 - eps)
        return -np.sum(np.sum(y * np.log(y_hat + 1e-5)))/m

    def backward(self, y, y_hat):
        eps = 1e-15
        y_hat = np.clip(y_hat, eps, 1 - eps)
        return (-(y / y_hat)).T
