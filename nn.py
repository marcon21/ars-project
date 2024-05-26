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
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))

        # Update state with the output of fc2 layer
        self.state = x.clone()

        x = F.softmax(self.fc3(x), dim=-1)

        vr = x[0].item()
        vl = x[1].item()

        return vr, vl

    def set_weights(self, state):
        """
        Set the weights and biases of the network from a list of 6 elements.
        The list should be in the following order: [fc1.weight, fc1.bias, fc2.weight, fc2.bias, fc3.weight, fc3.bias]
        """
        self.load_state_dict(state, strict=False)

    def get_weights(self):
        """
        Returns the weights of the network as a list of 6 elements
        the list is in the following order: [fc1.weight, fc1.bias, fc2.weight, fc2.bias, fc3.weight, fc3.bias]
        """
        return deepcopy(self.state_dict())
