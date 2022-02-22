import numpy as np


class Initializer:
    """
    Xavier: Tanh, Sigmoid
    He: ReLU, Leaky ReLU
    Resources: 
    https://stackoverflow.com/questions/48641192/xavier-and-he-normal-initialization-difference
    https://machinelearningmastery.com/weight-initialization-for-deep-learning-neural-networks/

    """

    def __call__(self):
        raise NotImplementedError('Must be implemented by the subclass')


class HeNormal(Initializer):
    """ He Usually used for ReLU type of activation function

    """

    def __call__(self, fan_in, fan_out):
        """ 
        Args:
            fan_in: number of input units. number of neurons for that layer
            fan-out: n, number of attributes

        Returns:
            np.arary initialized as specificed below

        Normal distribution, mean = 0, std = sqrt(2 / fan_in)
        """
        stddev = np.sqrt(2 / fan_in)
        return np.random.normal(0.0, stddev, (fan_in, fan_out))


class HeUniform(Initializer):
    """ Same as HeNormal but from a uniform distribution
    """

    def __call__(self, fan_in, fan_out):
        """
        Args: 
            fan_in: number of input units. number of neurons for that layer
            fan-out: n, number of attributes

        Returns: 
            np.arary initialized as specificed below

        From keras: 
        Uniform distribution [-limit, limit]
        Draws samples from a uniform distribution within [-limit, limit], 
        where limit = sqrt(6 / fan_in) 

        W: (fan_in, fan_out)
        """
        limit = np.sqrt(6 / fan_in)
        return np.random.uniform(-limit, limit, (fan_in, fan_out))


class XavierNormal(Initializer):
    def __call__(self, fan_in, fan_out):
        """
         Args: 
            fan_in: number of input units. number of neurons for that layer
            fan-out: n, number of attributes

        Returns: 
            np.arary initialized as specificed below

        Tanh or sigmoid
        Normal distribution, mean = 0, std = sqrt(2 / (fan_in + fan_out))
        """
        stddev = np.sqrt(2 / (fan_in + fan_out))
        return np.random.normal(0.0, stddev, (fan_in, fan_out))


class XavierUniform(Initializer):
    def __call__(self, fan_in, fan_out):
        """
         Args: 
            fan_in: number of input units. number of neurons for that layer
            fan-out: n, number of attributes

        Returns: 
            np.arary initialized as specificed below

        Used for Tanh or sigmoid
        From keras: 
        Draws samples from a uniform distribution within [-limit, limit], 
        where limit = sqrt(6 / (fan_in + fan_out)) 
        """
        limit = (np.sqrt(6) / (fan_in + fan_out))
        return np.random.uniform(-limit, limit, (fan_in, fan_out))
