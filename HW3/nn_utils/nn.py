import numpy as np
from sklearn.metrics import f1_score, accuracy_score
from sklearn.preprocessing import OneHotEncoder

from nn_utils.module import Module
from nn_utils.linear import Linear
from nn_utils.relu import ReLU
from nn_utils.sigmoid import Sigmoid
from nn_utils.softmax import Softmax
from nn_utils.tanh import Tanh
from nn_utils.dropout import Dropout
from nn_utils.loss import CrossEntropyLoss


def getActivationFunction(activationFunction):
    if activationFunction == "relu":
        return ReLU
    elif activationFunction == "sigmoid":
        return Sigmoid
    elif activationFunction == "tanh":
        return Tanh
    else:
        raise Exception("Invalid activation function")


#Model architecture
class Net(Module):
    def __init__(self, inputDim, hiddenDim, outputDim, numLayers, learningRate, activationFunction, dropoutRate=0.0):
        super(Net, self).__init__()

        self.fc1 = Linear(inputDim, hiddenDim, learningRate)
        self.afunc1 = activationFunction()

        self.fc_list = []
        self.afunc_list = []
        for i in range(numLayers-2):
            self.fc_list.append(Linear(hiddenDim, hiddenDim, learningRate))
            self.afunc_list.append(activationFunction())

        self.fcn = Linear(hiddenDim, outputDim, learningRate)
        self.afuncn = Softmax()

        self.dropout_layer = Dropout(dropoutRate)
        return
    
    def forward(self, x, train=True):
        x = self.afunc1.forward(self.fc1.forward(x))

        if train:
            x = self.dropout_layer.forward(x)

        for i in range(len(self.fc_list)):
            x = self.afunc_list[i].forward(self.fc_list[i].forward(x))
            if train:
                x = self.dropout_layer.forward(x)
    
        x = self.afuncn.forward(self.fcn.forward(x))
        return x
    
    def backward(self, gradients, train=True):
        gradients = self.afuncn.backward(gradients)
        gradients = self.fcn.backward(self.afuncn.backward(gradients))

        for i in range(len(self.fc_list)-1, -1, -1):
            if train:
                gradients = self.dropout_layer.backward(gradients)

            gradients = self.afunc_list[i].backward(gradients)
            gradients = self.fc_list[i].backward(gradients)

        if train:
            gradients = self.dropout_layer.backward(gradients)

        gradients = self.fc1.backward(self.afunc1.backward(gradients))

        #return gradients
    

    def step(self):
        self.fc1.step()
        for i in range(len(self.fc_list)):
            self.fc_list[i].step()
        self.fcn.step()
        return
    

    def num_params(self):
        num_params = self.fc1.num_params()
        for i in range(len(self.fc_list)):
            num_params += self.fc_list[i].num_params()
        num_params += self.fcn.num_params()
        return num_params
    

class Model:
    def __init__(self, **kwargs):
        self.inputDim = kwargs["inputDim"]
        self.hiddenDim = kwargs["hiddenDim"]
        self.outputDim = kwargs["outputDim"]
        self.numLayers = kwargs["numLayers"]
        self.learningRate = kwargs["learningRate"]
        self.activationFunction = getActivationFunction(kwargs["activationFunction"])
        self.dropoutRate = kwargs["dropoutRate"]
        self.model = Net(self.inputDim, self.hiddenDim, self.outputDim, self.numLayers, self.learningRate, self.activationFunction, self.dropoutRate)

        self.epochs = kwargs["epochs"]
        self.batchSize = kwargs["batchSize"]
        self.lossFunction = CrossEntropyLoss()

        self.trainLoss = []
        self.trainAccuracy = []
        self.valLoss = []
        self.valAccuracy = []

        self.mean = None
        self.std = None
        return
    

    def normalize(self, X, mean, std):
        if mean is None or std is None:
            mean = np.mean(X, axis=0)
            std = np.std(X, axis=0)

        return (X-mean)/std, mean, std


    def train_step(self, X, Y, train=True):
        #calculate the number of batches given the batch size
        numBatches = int(np.ceil(len(X)/self.batchSize))
        
        trainLoss = 0

        for batch in range(numBatches):
            #extracting the batch
            if batch == numBatches-1:
                x = X[int(self.batchSize*batch):,:]
                y = Y[int(self.batchSize*batch):]
            else:
                x = X[int(self.batchSize*batch):int(self.batchSize*(batch+1)),:]
                y = Y[int(self.batchSize*batch):int(self.batchSize*(batch+1))]

            #forward pass
            outputs = self.model.forward(x, train)
            loss = self.lossFunction.forward(outputs, y)
            trainLoss += loss

            #backward pass
            gradients = self.lossFunction.backward()
            self.model.backward(gradients, train)

            self.model.step()

        return trainLoss/numBatches
    

    def train(self, X_train, Y_train, X_val, Y_val, verbose=True, epochs_print=10, dropout=True):
        X_train_norm, self.mean, self.std = self.normalize(X_train, self.mean, self.std)
        X_val_norm, _, _ = self.normalize(X_val, self.mean, self.std)

        if Y_train.shape[1] != self.outputDim:
            self.encoder = OneHotEncoder(sparse_output=False)
            Y_train = self.encoder.fit_transform(Y_train)
            Y_val = self.encoder.transform(Y_val) 

        for epoch in range(self.epochs):
            trainLoss = self.train_step(X_train_norm, Y_train, train=dropout)
            valLoss = self.lossFunction.forward(self.model.forward(X_val, train=False), Y_val)

            if verbose:
                if (epoch+1)%epochs_print == 0:
                    print(f"Epoch: {epoch+1} | Train Loss: {trainLoss:.6f} | Val Loss: {valLoss:.6f} | Val Accuracy: {self.accuracy(X_val_norm, Y_val):.4f}"
                          f" | Val F1 Score: {self.f1_score(X_val_norm, Y_val):.4f}")

            self.trainLoss.append(trainLoss)
            self.valLoss.append(valLoss)

    
    def predict(self, X):
        X_norm, _, _ = self.normalize(X, self.mean, self.std)
        return self.model.forward(X_norm, train=False)
    

    def num_params(self):
        return self.model.num_params()
    

    def accuracy(self, X, Y):
        outputs = self.predict(X)
        predictions = np.argmax(outputs, axis=1)
        return accuracy_score(self.encoder.inverse_transform(Y), predictions)
    

    def f1_score(self, X, Y):
        outputs = self.predict(X)
        predictions = np.argmax(outputs, axis=1)
        #print(f"Y: {Y.shape} | Predictions: {predictions.shape}")
        return f1_score(self.encoder.inverse_transform(Y), predictions, average="micro")
