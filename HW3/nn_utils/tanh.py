import numpy as np
from nn_utils.module import Module

#definition for the Tanh layer class
class Tanh(Module):

    """
    Applies the Tanh function on each element of the input block.
    y = (exp(x)-exp(-x))/(exp(x)+exp(-x))

    Shapes:
    Input - (N,F)
    Output - (N,F)
    In Gradients - (N,F)
    Out Gradients - (N,F)
    """

    def __init__(self):
        super(Tanh, self).__init__()
        #declaring and initializing the cache dictionary
        self.cache = dict()
        self.cache["active"] = None
        return
    

    def forward(self, inputBatch):
        #applying the Tanh operation and storing a map showing where Tanh was active
        outputBatch = (np.exp(inputBatch)-np.exp(-inputBatch))/(np.exp(inputBatch)+np.exp(-inputBatch))
        self.cache["active"] = 1-outputBatch**2
        return outputBatch
    

    def backward(self, gradients):
        #computing the gradients wrt the input
        return gradients*self.cache["active"]
    