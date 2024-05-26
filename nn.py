from torch import nn
import torch
from torch.nn import functional as F
from copy import deepcopy
import numpy as np


class NN(nn.Module):
    def __init__(self, n_sensors=12, x1=32, x2=4, activation=F.relu):
        super(NN, self).__init__()
        self.fc1 = nn.Linear(n_sensors + x2, x1)
        self.fc2 = nn.Linear(x1, x2)
        self.fc3 = nn.Linear(x2, 2)

        self.state = torch.zeros(x2)
        self.activation = activation

    def forward(self, x):
        """
        Forward pass of the network
        """
        if torch.isnan(self.state).any():
            self.state = torch.zeros_like(self.state)

        x = torch.cat((x, self.state), dim=-1)
        x = self.activation(self.fc1(x))
        x = self.activation(self.fc2(x))

        # Update state with the output of fc2 layer
        self.state = x.clone()

        x = F.softmax(self.fc3(x), dim=-1)

        vr = x[0].item()
        vl = x[1].item()

        assert np.isnan(vr) == False, "NaN in the network vr"
        assert np.isnan(vl) == False, "NaN in the network vl"
        return vr, vl

    def set_weights(self, weights):
        """
        Set the weights and biases of the network from a list of 6 elements.
        The list should be in the following order: [fc1.weight, fc1.bias, fc2.weight, fc2.bias, fc3.weight, fc3.bias]
        """
        if len(weights) != 6:
            raise ValueError("Weights should be a list of 6 elements")
        self.fc1.weight.data = weights[0]
        self.fc1.bias.data = weights[1]
        self.fc2.weight.data = weights[2]
        self.fc2.bias.data = weights[3]
        self.fc3.weight.data = weights[4]
        self.fc3.bias.data = weights[5]

    def get_weights(self):
        """
        Returns the weights of the network as a list of 6 elements
        the list is in the following order: [fc1.weight, fc1.bias, fc2.weight, fc2.bias, fc3.weight, fc3.bias]
        """
        return [
            self.fc1.weight.data,
            self.fc1.bias.data,
            self.fc2.weight.data,
            self.fc2.bias.data,
            self.fc3.weight.data,
            self.fc3.bias.data,
        ]
