from torch import nn
import torch
from torch.nn import functional as F

class NN(nn.Module):
    def __init__(self, n_sensors=12, x1=32, x2=4, activation=F.relu):
        super(NN, self).__init__()
        self.fc1 = nn.Linear(n_sensors+x2, x1)
        self.fc2 = nn.Linear(x1, x2)
        self.fc3 = nn.Linear(x2, 2)
        self.activation = activation

    def forward(self, x):
        x = self.activation(self.fc1(x))
        x = self.activation(self.fc2(x))
        x = F.softmax(self.fc3(x))
        return x
    
    def set_weights(self, weights):
        if len(weights) != 6 :
            raise ValueError("Weights should be a list of 6 elements")
        self.fc1.weight.data = weights[0]  
        self.fc1.bias.data = weights[1]
        self.fc2.weight.data = weights[2]
        self.fc2.bias.data = weights[3]
        self.fc3.weight.data = weights[4]
        self.fc3.bias.data = weights[5]
        
    def get_weights(self):
        return [self.fc1.weight.data, self.fc1.bias.data, self.fc2.weight.data, self.fc2.bias.data, self.fc3.weight.data, self.fc3.bias.data]