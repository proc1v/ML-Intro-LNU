import numpy as np

#the base class for any layer or module
class Module:
    def __init__(self):
        return
    
    def forward(self, inputBatch):
        return inputBatch
    
    def backward(self, gradients):
        g = np.eye(gradients.shape[1])
        return np.dot(gradients, g)