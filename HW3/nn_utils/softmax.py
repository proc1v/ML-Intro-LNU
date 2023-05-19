import numpy as np
from nn_utils.module import Module

#definition for the Softmax layer class    
class Softmax(Module):
    
    """
    Applies the Softmax function along the last axis/dimension of the input batch
    yi = exp(xi)/sum_j(exp(xj))
    
    Shapes:
    Input - (N,F)
    Output - (N,F)
    In Gradients - (N,F)
    Out Gradients - (N,F)
    """
    
    def __init__(self):
        super(Softmax, self).__init__()
        #declaring and initializing the cache dictionary
        self.cache = dict()
        self.cache["output"] = None
        return
    
    
    def forward(self, inputBatch):
        #computing the exponential values along with a correction factor for numerical stability
        correction = np.max(inputBatch, axis=1, keepdims=True)
        expVals = np.exp(inputBatch - correction)
        
        #computing the softmax operation
        outputBatch = expVals/np.sum(expVals, axis=1, keepdims=True)
        
        #storing the outputs in the cache
        self.cache["output"] = outputBatch
        return outputBatch
    

    def backward(self, gradients):
        #computing the gradients wrt the input
        [N,F] = self.cache["output"].shape
        delta = np.eye(F,F)
        delta = np.reshape(delta, (1,F,F))
        output = np.reshape(self.cache["output"], (N,1,F))
        inGrad = output.transpose(0,2,1)*(delta - output)
        return np.einsum('nf,nfd->nd', gradients, inGrad)
    