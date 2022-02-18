import numpy as np

"""
Xavier: Tanh, Sigmoid
He: ReLU, Leaky ReLU
Resources: 
https://stackoverflow.com/questions/48641192/xavier-and-he-normal-initialization-difference
https://machinelearningmastery.com/weight-initialization-for-deep-learning-neural-networks/

"""


def he_normal(fan_in, fan_out):
    """
    fan_in: number of input units. number of neurons for that layer
    fan-out: n, number of attributes
    W: (fan_in, fan_out)

    Normal distribution, mean = 0, std = sqrt(2 / fan_in)
    """
    stddev = np.sqrt(2 / fan_in)
    return np.random.normal(0.0, stddev, (fan_in, fan_out))


def he_uniform(fan_in, fan_out):
    """
    fan_in: number of input units. number of neurons for that layer
    fan-out: n, number of attributes
    W: (fan_in, fan_out)

    From keras: 
    Uniform distribution [-limit, limit]
    Draws samples from a uniform distribution within [-limit, limit], 
    where limit = sqrt(6 / fan_in) 
    """
    limit = np.sqrt(6 / fan_in)
    return np.random.uniform(-limit, limit, (fan_in, fan_out))


def xavier_normal(fan_in, fan_out):
    """
    Tanh or sigmoid
    Normal distribution, mean = 0, std = sqrt(2 / (fan_in + fan_out))
    """
    stddev = np.sqrt(2 / (fan_in + fan_out))
    return np.random.normal(0.0, stddev, (fan_in, fan_out))


def xavier_uniform(fan_in, fan_out):
    """
    Used for Tanh or sigmoid

    From keras: 
    Draws samples from a uniform distribution within [-limit, limit], 
    where limit = sqrt(6 / (fan_in + fan_out)) 
    """
    limit = (np.sqrt(6) / (fan_in + fan_out))
    return np.random.uniform(-limit, limit, (fan_in, fan_out))
