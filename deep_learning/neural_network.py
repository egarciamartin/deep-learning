import numpy as np
from sklearn.metrics import accuracy_score
from deep_learning import layers
from deep_learning.losses import MAE, CrossEntropy
from deep_learning.optimizers import SGD


class NeuralNetwork:
    def __init__(self, optimizer, loss) -> None:
        self.layers = list()
        self.optimizer = optimizer
        self.loss = loss

    def add(self, layer):
        # Passing the optimizer to the layer
        if layer.optimizer == None:
            layer.optimizer = self.optimizer
        self.layers.append(layer)

    def forward(self, x):
        """
        Forward pass
        Starting from the input layer, calculate the W matrices, bias 

        Args
            x: input data. shape: (m, n)
        Returns 
            output.T: So that it's in the expected shape: (neurons, m)
        """
        output = x.T
        for layer in self.layers:
            output = layer.forward(output)

        return output.T

    def fit(self, x, y, epochs, verbose=1):
        """
        Imagine:
        Input, FC, FC, Output

        First, all data one batch
        x = (m, n)
        m can be split in batches

        """

        for i in range(epochs):
            y_hat = self.forward(x)
            l = self.loss.loss(y, y_hat)
            self.backprop(y, y_hat)
            if verbose >= 1:
                print(f'Epoch: {i+1} : {l:.3f}')
        print(f"Loss after {epochs} epochs {l:.3f}")

    def backprop(self, y, y_hat) -> None:
        """
        Backward pass: back propagation
        Resources:
        http://neuralnetworksanddeeplearning.com/chap2.html
        https://www.deeplearningbook.org/contents/mlp.html 
        https://www.ics.uci.edu/~pjsadows/notes.pdf

        1. dloss from the loss function. That is the incoming gradient for the previous layer
        2. next step is to derive the activation function, and pass back through the z operation 
        on W.X + b. 
        3. The new dloss is passed back to the previous layer, through the incoming connection. 

        So:
        1. dloss from the loss function
        2. For every layer: 
         1. delta = dloss * activation.backward
         2. db = delta
         3. dw = delta * (incoming a to that layer, output of the previous layer in forward)
         4. dloss = delta * W.T   

        Softmax: It's easier to calculate the dloss from the equation of the softmax 
        together with the derivative of the cross entropy. 
        So we do that at the same time. Exception for softmax
        """
        if self.layers[-1].activation.name == 'softmax':
            dloss = self.layers[-1].activation.backward(y, y_hat)
        else:
            dloss = self.loss.backward(y, y_hat)
        for layer in reversed(self.layers[1:]):
            # Operations performed at the layer level
            dloss = layer.backward(dloss)

    def predict(self, x):
        y_hat = self.forward(x)
        return y_hat

    def evaluate(self, x, y, metric='accuracy'):
        y_pred = self.predict(x)
        if metric == 'accuracy':
            return accuracy_score(np.argmax(y, axis=1), np.argmax(y_pred, axis=1))
        else:
            raise Exception("Metric not implemented")

    def summary(self):
        print(self.layers)
