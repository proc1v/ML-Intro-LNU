import numpy as np
from nn_utils.module import Module

#definition for the Sigmoid layer class
class Sigmoid(Module):

    """
    Applies the Sigmoid function on each element of the input block.
    y = 1/(1+exp(-x))

    Shapes:
    Input - (N,F)
    Output - (N,F)
    In Gradients - (N,F)
    Out Gradients - (N,F)
    """

    def __init__(self):
        super(Sigmoid, self).__init__()
        #declaring and initializing the cache dictionary
        self.cache = dict()
        self.cache["active"] = None
        return
    

    def forward(self, inputBatch):
        #applying the Sigmoid operation and storing a map showing where Sigmoid was active
        outputBatch = 1/(1+np.exp(-inputBatch))
        self.cache["active"] = outputBatch*(1-outputBatch)
        return outputBatch
    

    def backward(self, gradients):
        #computing the gradients wrt the input
        return gradients*self.cache["active"]
    