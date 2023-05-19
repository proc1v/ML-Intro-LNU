import numpy as np

def linear(x):
    return x

def linear_derivative(x):
    return np.ones_like(x)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return sigmoid(x) * (1 - sigmoid(x))

def softmax(x):
    exps = np.exp(x - np.max(x, axis=1, keepdims=True))
    return exps / np.sum(exps, axis=1, keepdims=True)

def tanh(x):
    return np.tanh(x)

def tanh_derivative(x):
    return 1 - np.tanh(x)**2

def relu(x):
    return np.maximum(0, x)

def relu_derivative(x):
    return np.where(x > 0, 1, 0)

def activation_function(x, activation):
    if activation == 'sigmoid':
        return sigmoid(x)
    elif activation == 'tanh':
        return tanh(x)
    elif activation == 'relu':
        return relu(x)
    elif activation == 'linear':
        return linear(x)
    
def activation_derivative(x, activation):
    if activation == 'sigmoid':
        return sigmoid_derivative(x)
    elif activation == 'tanh':
        return tanh_derivative(x)
    elif activation == 'relu':
        return relu_derivative(x)
    elif activation == 'linear':
        return linear_derivative(x)