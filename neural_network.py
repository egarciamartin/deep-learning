import numpy as np
from sklearn import datasets
import layers


class NeuralNetwork:
    def __init__(self, optimizer, loss) -> None:
        self.layers = list()
        self.optimizer = optimizer
        self.loss = loss

    def add(self, layer):
        self.layers.append(layer)

    def forward(self, x):
        """
        Forward pass
        Starting from the input layer, calculate the W matrices, bias 
        Transposing it because Tensorflow uses: x: (m, n) but to do the matrix multipl
        we need x.T
        """
        output = x.T
        for layer in self.layers:
            print(layer.name)
            output = layer.forward(output)

        return output

    def fit(self, x, y, epochs):
        """
        Imagine:
        Input, FC, FC, Output

        First, all data one batch
        input = (m, dim1, dim2)
        m can be split in batches

        """
        for i in range(epochs):
            # 1. Forward pass
            y_hat = self.forward(x)
            print(f"{i} y_hat, output of the last layer:\n {y_hat}")
            # 2. Loss
            l = self.calculate_loss(y, y_hat)
            # 3.Backprop
            self.backprop(l)

    def calculate_loss(self, y, y_hat):

        return self.loss(y, y_hat)

    def backprop(self, l):
        pass

    def summary(self):
        print(self.layers)


data = datasets.load_digits()
x = data.data
y = data.target

# opt = Adam()
# loss = CrossEntropyLoss()
model = NeuralNetwork("opt", "loss")
model.add(layers.Input(input_shape=(x.shape)))
model.add(layers.Dense(8, activation='relu'))
model.add(layers.Dense(8, activation='relu'))
model.add(layers.Dense(2, activation='sigmoid'))
model.summary()
model.fit(x, y, 10)

print(model.layers[0].input_shape)
print(model.layers[1].input_shape)
