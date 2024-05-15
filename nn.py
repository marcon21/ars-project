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
    
    def set_weights(self,weights):
        pass