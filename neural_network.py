import numpy as np


class NeuralNetwork:
    """ A neural network of L number of layers
    Based on the code from deeplearning.ai

    Parameters:
    layers_dims : list with the dimension of each layer
    """

    def __init__(self,
                 layers_dims,
                 epochs,
                 learning_rate,
                 ):
        self.layers_dims = layers_dims
        self.epochs = epochs
        self.learning_rate = learning_rate

    def initialize_parameters(self):
        """
        Returns:
        parames : dictionary
            W weights and b biases
        """
        params = dict()
        L = len(self.layers_dims)
        for l_ in range(1, L):
            # The 0 layer is the input layer
            params['W' + str(l_)] = np.random.randn(self.layers_dims[l_],
                                                    self.layers_dims[l_ - 1]) * 0.01
            params['b' + str(l_)] = np.zeros((self.layers_dims[l_], 1))
        return params

    def forward_prop(self, X, params):
        """Forward propagation
        First all the RELU layers, last the sigmoid layer
        For each layer, calculate the W, b, Z, A matrices
        Those will be saved in cache to be able to compute backprop (gradients)

        Arguments:
        X -- data, numpy array of shape (input size, number of examples)
        parameters -- output of initialize_parameters_deep()

        Returns:
        AL -- last post-activation value
        caches -- list of caches containing:
                    every cache of linear_activation_forward()
                    (there are L-1 of them, indexed from 0 to L-1)
        """
        caches = list()
        A = X
        L = len(params) // 2
        # RELU layers
        for l_ in range(1, L):
            A_prev = A
            W = params['W' + str(l_)]
            b = params['b' + str(l_)]
            Z = np.dot(W, A_prev) + b
            A, activation_cache = self.relu(Z)
            caches.append((A, W, b, Z))

        # Latest SIGMOID layer
        W = params['W' + str(L)]
        b = params['b' + str(L)]
        Z = np.dot(W, A) + b
        AL, activation_cache = self.sigmoid(Z)
        caches.append((AL, W, b, Z))

        return AL, caches

    def backward_layer(self, cache, activation, dA, layer, grads):
        """Backward prop for one layer
        """
        A_prev, W, b, Z = cache
        m = A_prev.shape[1]
        if activation == "sigmoid":
            dZ = self.d_sigmoid(dA, Z)
        elif activation == "relu":
            dZ = self.d_relu(dA, Z)

        grads["dA" + str(layer)] = np.dot(W.T, dZ)
        grads["dW" + str(layer+1)] = 1 / m * np.dot(dZ, A_prev.T)
        grads["db" + str(layer+1)] = 1 / m * np.sum(dZ, axis=1, keepdims=True)

        return grads

    def backward_prop(self, AL, Y, caches):
        grads = dict()
        L = len(caches)
        cache = caches[-1]
        dAL = - (np.divide(Y, AL) - np.divide(1 - Y, 1 - AL))
        grads = self.backward_layer(cache, 'sigmoid', dAL, L-1)
        # A_prev, W, b, Z = cache
        # m = A_prev.shape[1]
        # Lth layer. Input: dAL, current cache. Output: dW, db, dAL-1
        # derivative of cost with respect to AL


        # dZ = self.d_sigmoid(dAL, Z)

        # linear_cache = A_prev, W, b
        # grads["dA" + str(L - 1)] = np.dot(W.T, dZ)
        # grads["dW" + str(L)] = 1 / m * np.dot(dZ, A_prev.T)
        # grads["db" + str(L)] = 1 / m * np.sum(dZ, axis=1, keepdims=True)

        for i in reversed(range(L - 1)):
            cache = caches[i]
            dA = grads["dA" + str(i - 1)]
            grads = self.backward_layer(cache, 'relu', dA, i, grads)
            # A_prev, W, b, Z = cache
            # dZ = self.d_relu(grads["dA" + str(i - 1)], Z)
            # grads["dA" + str(i)] = np.dot(W.T, dZ)
            # grads["dW" + str(i + 1)] = 1 / m * np.dot(dZ, A_prev.T)
            # grads["db" + str(i + 1)] = 1 / m * \
            #     np.sum(dZ, axis=1, keepdims=True)
        return grads

    def calculate_cost(self, AL, Y):
        m = Y.shape[1]
        cost = - 1 / m * np.sum(Y * np.log(AL) + (1 - Y) * np.log(1 - AL))
        cost = np.squeeze(cost)  # make sure shape is correct

        return cost

    def update_parameters(self, params, grads, lr):
        """ params = W, b (dict)
            grads =
            W = W - lr.dW
            b = b - lr.db
        """
        parameters = params.copy()
        L = len(parameters) // 2
        for i in range(L):
            W = params['W' + str(i + 1)]
            dW = grads['dW' + str(i + 1)]
            b = params['b' + str(i + 1)]
            db = grads['db' + str(i + 1)]
            parameters['W' + str(i + 1)] = W - (lr * dW)
            parameters['b' + str(i + 1)] = b - (lr * db)
        return parameters

    def sigmoid(self, Z):
        return 1 / (1 + np.exp(-Z))

    def relu(self, Z):
        return np.maximum(0, Z)

    def d_sigmoid(self, dA, Z):
        sig = self.sigmoid(Z)
        dZ = dA * (sig * (1 - sig))
        return dZ

    def d_relu(dA, Z):
        """ Create an array with dA, since we have to multiply it later, we
        save that step
        """
        dZ = np.array(dA, copy=True)
        dZ[Z < 0] = 0.0
        return dZ

    def train(self, X, Y):
        """Training the neural network.
        1. Initialize the parameters
        2. Forward propagation
        3. Calculate the cost
        4. Backward propagation
        5. Update the parameters
        Repeat this for a number of epochs

        Arguments
        ---------
        X : data, numpy array of shape
            The training data
        Y : data, numpy array of shape (1, number of instances)
            The labeled data
        """
        params = self.initialize_parameters()
        costs = list()

        for i in range(0, self.epochs):
            AL, caches = self.forward_prop(X, params)
            cost = self.calculate_cost(AL, Y)
            grads = self.backward_prop(AL, Y, caches)
            params = self.update_parameters(params, grads, self.learning_rate)
            costs.append(cost)
        return params, costs
