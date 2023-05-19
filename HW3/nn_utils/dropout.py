import numpy as np
from nn_utils.module import Module

#definition for the Dropout layer class
class Dropout(Module):

    """
    Applies the Dropout function on each element of the input block.
    y = x with probability p
    y = 0 with probability 1-p
    
    Shapes:
    Input - (N,F)
    Output - (N,F)
    In Gradients - (N,F)
    Out Gradients - (N,F)
    """

    def __init__(self, p):
        super(Dropout, self).__init__()
        #declaring and initializing the cache dictionary
        self.cache = dict()
        self.cache["active"] = None
        self.p = p
        return
    

    def forward(self, inputBatch):
        #applying the Dropout operation and storing a map showing where Dropout was active
        self.cache["active"] = np.random.binomial(1, self.p, size=inputBatch.shape)
        return inputBatch*self.cache["active"]
    

    def backward(self, gradients):
        #computing the gradients wrt the input
        return gradients*self.cache["active"]
    