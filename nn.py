from torch import nn
import torch
from torch.nn import functional as F


class NN(nn.Module):
    '''
    Neural network class
    
    Attributes:
        n_sensors: int, number of sensors in the input
        x1: int, number of neurons in the first hidden layer
        x2: int, number of neurons in the second hidden layer
        activation: function, activation function of the hidden layers
        fc1: nn.Linear, first hidden layer  
        fc2: nn.Linear, second hidden layer
        fc3: nn.Linear, output layer
    '''
    def __init__(self, n_sensors=12, x1=32, x2=4, activation=F.relu):
        super(NN, self).__init__()
        #self.fc1 = nn.Linear(n_sensors+x2, x1)  TOADD
        self.fc1 = nn.Linear(n_sensors, x1)  #TOREMOVE
        self.fc2 = nn.Linear(x1, x2)
        self.fc3 = nn.Linear(x2, 2)
        self.activation = activation

    def forward(self, x):
        '''
        Forward pass of the network
        '''
        if not isinstance(x, torch.Tensor):
            raise ValueError("Input should be a torch.Tensor")
        self.fc1.weight.data = torch.randn_like(self.fc1.weight.data) #TOREMOVE
        self.fc2.weight.data = torch.randn_like(self.fc2.weight.data) #TOREMOVE
        self.fc3.weight.data = torch.randn_like(self.fc3.weight.data) #TOREMOVE
        self.fc1.bias.data = torch.randn_like(self.fc1.bias.data)  #TOREMOVE
        self.fc2.bias.data = torch.randn_like(self.fc2.bias.data) #TOREMOVE
        self.fc3.bias.data = torch.randn_like(self.fc3.bias.data) #TOREMOVE
        x = self.activation(self.fc1(x))
        x = self.activation(self.fc2(x))
        vr = x[0].item()
        vl = x[1].item()
        return vl, vr
    
    def set_weights(self, weights):
        '''
        Set the weights and biases of the network from a list of 6 elements.
        The list should be in the following order: [fc1.weight, fc1.bias, fc2.weight, fc2.bias, fc3.weight, fc3.bias]
        '''
        if len(weights) != 6 :
            raise ValueError("Weights should be a list of 6 elements")
        self.fc1.weight.data = weights[0]  
        self.fc1.bias.data = weights[1]
        self.fc2.weight.data = weights[2]
        self.fc2.bias.data = weights[3]
        self.fc3.weight.data = weights[4]
        self.fc3.bias.data = weights[5]
        
    def get_weights(self):
        '''
        Returns the weights of the network as a list of 6 elements
        '''
        return [self.fc1.weight.data, self.fc1.bias.data, self.fc2.weight.data, self.fc2.bias.data, self.fc3.weight.data, self.fc3.bias.data]