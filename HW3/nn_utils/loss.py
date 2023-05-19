import numpy as np
from nn_utils.module import Module

#definition for the Cross Entropy Loss layer class
class CrossEntropyLoss(Module):
    
    """
    Computes the cross entropy loss using the output and the required target class.
    loss = -log(yHat[class])
    
    Shapes:
    Outputs - (N,C)
    Classes - (N)
    Out Gradients - (N,C)
    """
    
    def __init__(self):
        super(CrossEntropyLoss, self).__init__()
        #declaring and initializing the cache dictionary
        self.cache = dict()
        self.cache["scores"] = None
        self.cache["classes"] = None
        self.cache["numClasses"] = None
        return
    
    
    def forward(self, y_pred, y_true):
        #computing the loss
        eps = np.finfo(float).eps
        cross_entropy = -np.sum(y_true * np.log(y_pred + eps)) / y_pred.shape[0]

        #storing the outputs and the classes in the cache
        self.cache["scores"] = y_pred
        self.cache["classes"] = y_true
        self.cache["numClasses"] = y_pred.shape[1]
        return cross_entropy

    
    def backward(self):
        grad = self.cache["scores"] - self.cache["classes"]
        return grad
    