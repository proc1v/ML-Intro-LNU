import numpy as np
from nn_utils.module import Module

#definition for the ReLU layer class
class ReLU(Module):
    
    """
    Applies the ReLU function on each element of the input block.
    y = max(0,x)
    
    Shapes:
    Input - (N,F)
    Output - (N,F)
    In Gradients - (N,F)
    Out Gradients - (N,F)
    """
    
    def __init__(self):
        super(ReLU, self).__init__()
        #declaring and initializing the cache dictionary 
        self.cache = dict()
        self.cache["active"] = None
        return
    
    
    def forward(self, inputBatch):
        #applying the ReLU operation and storing a map showing where ReLU was active
        outputBatch = np.maximum(0, inputBatch)
        self.cache["active"] = (inputBatch > 0)
        return outputBatch
    

    def backward(self, gradients):
        #computing the gradients wrt the input
        inGrad = np.zeros(self.cache["active"].shape)
        inGrad[self.cache["active"]] = 1
        return gradients*inGrad
    