import numpy as np
from deep_learning.activation_functions import LeakyReLU, ReLU, Tanh, Sigmoid, Softmax, Linear
from deep_learning.initializers import HeNormal, XavierNormal, HeUniform, XavierUniform


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
        self.z = None
        # outcoming a
        self.a = None
        self.input_shape = None

        # Incoming input from previous layer in forward pass
        self.incoming_a = None

        # Optimizer
        self.optimizer = None

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
    z: (neurons, m)

    z = W.X + b
    a = f(z) where f is the activation function
    """

    def __init__(self, neurons, activation, initializer=None) -> None:
        super().__init__()
        self.neurons = neurons
        self.name = "Dense"
        self.initializer = initializer

        # Activation function setup
        if activation == 'relu':
            self.activation = ReLU()
        elif activation == 'tanh':
            self.activation = Tanh()
        elif activation == 'sigmoid':
            self.activation = Sigmoid()
        elif activation == 'leaky_relu':
            self.activation = LeakyReLU()
        elif activation == 'linear':
            self.activation = Linear()
        elif activation == 'softmax':
            self.activation = Softmax()
        else:
            raise Exception(
                f"Activation function {self.activation} not supported")

    def initialize(self, input_shape):
        """
        Calling different weight initializers
        ReLU: He 
        Tanh, Sigmoid: Xavier

        bias initialized to zero
        input_shape: (n, m) it has been transposed
        """
        self.input_shape = input_shape
        if self.initializer is None:
            if self.activation.name in {"relu", 'leaky_relu', 'linear', 'softmax'}:
                self.initializer = HeNormal()
            else:
                self.initializer = XavierNormal()

        self.W = self.initializer(self.neurons, input_shape[0])
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

        self.incoming_a = input
        self.z = np.dot(self.W, input) + self.b
        self.a = self.activation(self.z)

        return self.a

    def backward(self, dloss):
        """
        Getting the dloss as input (incoming gradient) from the previous layer (l+1)

        For every layer: 
         1. delta = dloss * activation.backward(of a, the output)
         2. db = delta
         3. dw = delta * (incoming a to that layer, output of the previous layer in forward)
         4. dloss = delta * W.T   
        """

        if self.activation.name == 'softmax':
            delta = dloss
        else:
            delta = dloss * self.activation.backward(self.a, self.z)
        db = delta
        dW = np.dot(delta, self.incoming_a.T)
        dloss = np.dot(self.W.T, delta)

        self.W = self.optimizer.opt(self.W, dW)
        self.b = self.optimizer.opt(self.b, db)

        return dloss
