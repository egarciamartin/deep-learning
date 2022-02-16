import numpy as np


class Loss:
    def __init__(self) -> None:
        pass


class MAE(Loss):
    def __init__(self) -> None:
        super().__init__()

    def loss(y, y_hat):
        """
        Returns the MAE between y and y_hat
        """
        pass

    def backward():
        pass


class CrossEntropy(Loss):
    def loss(y, y_hat):
        pass

    def backward():
        pass
