from env import Enviroment
from actors import Agent
import numpy as np
import torch
import torch.nn as nn
from nn import NN
from torch.autograd import Variable
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
import pygad


class Evolution:

    def __init__(
        self,
        env: Enviroment,
        input_dim,
        hidden_dim,
        layer_dim,
        output_dim,
        initial_population_size,
    ):

        self.env = env
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.layer_dim = layer_dim
        self.output_dim = output_dim
        self.initial_population_size = initial_population_size

    def create_model(self):

        return NN(self.input_dim, self.hidden_dim, self.output_dim)

    def create_genetic_representation(self, model):

        representation = []
        for p in model.parameters():
            representation.append(p.data.numpy().flatten())

        return representation

    def create_population(self):
        self.population = []
        Weights = []
        for i in range(self.initial_population_size):
            x, y = np.random.randint(0, self.env.width), np.random.randint(
                0, self.env.height
            )

            agent = Agent(
                x=x,
                y=y,
                move_speed=5,
                size=40,
                n_sensors=10,
                max_distance=200,
                color="red",
                model=self.create_model(),
            )

            # agent.model.init_hidden()
            if i == 0:
                number_par = (
                    (self.input_dim * self.hidden_dim)
                    + self.hidden_dim
                    + (self.hidden_dim * self.hidden_dim)
                    + self.hidden_dim
                    + (self.hidden_dim * self.output_dim)
                    + self.output_dim
                )
                weight = np.random.randn(number_par)
                Weights.append(weight)
                for n, p in zip(weight, self.agent.model.parameters()):
                    p.data += torch.FloatTensor(weight)

            else:

                prec_weight = Weights[-1]
                variance_weights = np.full(number_par, 0.1)
                weight = np.random.normal(loc=prec_weight, scale=variance_weights)
                for n, p in zip(weight, self.agent.model.parameters()):
                    p.data += torch.FloatTensor(weight)

            self.population.append(agent)

    def fitness(self, agent: Agent):

        pass

    def selection(self):
        pass
