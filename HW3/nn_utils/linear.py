import numpy as np
from nn_utils.module import Module

#definition for the Linear layer class
class Linear(Module):
    
    """
    Fully Connected layer that applies a linear transformation to the input using weights and biases.
    y = w.x + b
    
    Shapes:
    Input - (N, Fin)
    Output - (N, Fout)
    In Gradients - (N, Fout)
    Out Gradients - (N, Fin)
    """
    
    def __init__(self, inFeatures, outFeatures, learningRate):
        super(Linear, self).__init__()
        self.inFeatures = inFeatures
        self.outFeatures = outFeatures
        self.lr = learningRate
        
        #declaring the weight and bias dictionaries for storing their values and gradients
        self.weight = dict()
        self.bias = dict()
        
        #initializing the weight and bias values and gradients
        self.weight["grad"] = None
        self.bias["grad"] = None
        self.weight["val"] = np.random.uniform(-np.sqrt(1/inFeatures), np.sqrt(1/inFeatures), size=(outFeatures, inFeatures))
        self.bias["val"] = np.random.uniform(-np.sqrt(1/inFeatures), np.sqrt(1/inFeatures), size=(outFeatures))
        #declaring and initializing the cache dictionary
        self.cache = dict()
        self.cache["input"] = None
        return
    
    
    def forward(self, inputBatch):
        #computing the linear transformation and storing the inputs in the cache
        outputBatch = np.dot(inputBatch, self.weight["val"].T) + self.bias["val"]
        self.cache["input"] = inputBatch
        return outputBatch
    
    
    def backward(self, gradients):
        #computing the gradients wrt the weight
        [N, Fin] = self.cache["input"].shape
        wGrad = np.einsum('no,ni->noi', gradients, self.cache["input"])
        self.weight["grad"] = np.mean(wGrad, axis=0)
        
        #computing the gradients wrt the bias
        bGrad = np.dot(gradients, np.eye(gradients.shape[1]))
        self.bias["grad"] = np.mean(bGrad, axis=0)
        
        #computing the gradients wrt the input
        inGrad = self.weight["val"]
        return np.dot(gradients, inGrad)
    
    
    def step(self):
        #weight and bias values update
        self.weight["val"] = self.weight["val"] - self.lr*self.weight["grad"]
        self.bias["val"] = self.bias["val"] - self.lr*self.bias["grad"]
        return
    
    
    def num_params(self):
        #total number of trainable parameters in the layer
        numParams = (self.weight["val"].shape[0]*self.weight["val"].shape[1]) + self.bias["val"].shape[0]
        return numParams