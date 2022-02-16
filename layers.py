from ast import Raise
import numpy as np
from activation_functions import leaky_relu, relu, tanh, sigmoid
from initializers import he_normal, xavier_normal


class Layers:
    """
    Input layer: saves the dimensions
    Dense layer: fully-connected layer
        - Activation function
        - W: weight matrix
        - b: bias vector

    """

    def __init__(self) -> None:
        self.W = None
        self.b = None
        self.input_shape = None

    def forward(self, input):
        pass

    def backward(self):
        pass


class Input(Layers):
    """
    Shapes:
    - X: (m, n), for easier use for the batch.
    But it's confusing when accessing n, and doing transposes. 
    So the input, the first round will be .T, then everything is in order
    """

    def __init__(self, input_shape) -> None:
        super().__init__()
        self.input_shape = input_shape
        self.name = "Input"

    def forward(self, input):
        return input


class Dense(Layers):
    """
    input_shape (from super()): shape of X: (m, n)
    Shapes: 
    W: (neurons, prev_layer n)
    b: (neurons, 1)

    """

    def __init__(self, neurons, activation) -> None:
        super().__init__()
        self.neurons = neurons
        self.activation = activation
        self.name = "Dense"

    def initialize(self, input_shape):
        """
        Calling different weight initializers
        ReLU: He 
        Tanh, Sigmoid: Xavier

        bias initialized to zero
        input_shape: (n, m) it has been transposed
        """
        self.input_shape = input_shape
        if self.activation == "relu" or self.activation == 'leaky_relu':
            self.W = he_normal(self.neurons, input_shape[0])
        elif self.activation == "tanh" or self.activation == "sigmoid":
            self.W = xavier_normal(self.neurons, input_shape[0])
        else:
            raise Exception(
                f"Activation function {self.activation} not supported")

        self.b = np.zeros((self.neurons, 1))

    def forward(self, input):
        """
        input: input to that layer, output from the previous layer
        1. First pass: Initialize W and b matrices 
        2. z = W.X + b
        3. output = activation(z)
        """
        if self.input_shape is None:
            # First pass
            self.initialize(input.shape)

        z = np.dot(self.W, input) + self.b

        if self.activation == 'relu':
            output = relu(z)
        elif self.activation == 'tanh':
            output = tanh(z)
        elif self.activation == 'sigmoid':
            output = sigmoid(z)
        elif self.activation == 'leaky_relu':
            output = leaky_relu(z)
        else:
            raise Exception(
                f"Activation function {self.activation} not supported")
        return output
