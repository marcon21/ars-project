from env import Enviroment
from actors import Agent
from nn import NN
from utils import create_pairs
import math
import numpy as np
import torch


# Author: Aurora Pia Ghiardelli
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
            )
            self.agent.model = self.create_model()
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

            self.agent.genetic_representation = self.create_genetic_representation(
                self.agent.model
            )
            self.population.append(agent)

    def proportionate_selection(
        self,
    ):  # Selection of the agents based on their fitness score with the proportionate selection method

        fitness_scores = [agent.fitness_score for agent in self.population]
        sum_fitness = sum(fitness_scores)
        probabilities = [score / sum_fitness for score in fitness_scores]
        self.population = np.random.multinomial(
            len(self.population), probabilities, size=None
        )  # multinomial distribution where each agent is selected with a probability proportional to its fitness score

    def rank_based_selection(
        self,
    ):  # Selection of the agents based on their fitness score with the rank based selection method

        sorted_population = sorted(
            self.population, key=lambda x: x.fitness_score, reverse=True
        )
        probabilities = [1 / (i + 1) for i in range(len(sorted_population))]
        probabilities /= np.sum(probabilities)
        self.population = np.random.multinomial(
            len(self.population), probabilities, size=None
        )

    def crossover(self):

        pairs = create_pairs(self.population, fertility_rate=0.5)
        new_population = []
        for pair in pairs:
            parent1, parent2 = pair
            child1 = Agent(
                x=parent1.pos[0],
                y=parent1.pos[1],
                move_speed=5,
                size=40,
                n_sensors=10,
                max_distance=200,
                color="red",
            )
            child2 = Agent(
                x=parent2.pos[0],
                y=parent2.pos[1],
                move_speed=5,
                size=40,
                n_sensors=10,
                max_distance=200,
                color="red",
            )
            for i, (p1, p2, c1, c2) in enumerate(  # Crossover between the two parents
                zip(
                    parent1.genetic_representation,
                    parent2.genetic_representation,
                    child1.genetic_representation,
                    child2.genetic_representation,
                )
            ):
                alpha = np.random.rand()  # Random number between 0 and 1
                c1 = (
                    alpha * p1 + (1 - alpha) * p2
                )  # Linear combination of the two parents' genetic representations
                c2 = alpha * p2 + (1 - alpha) * p1
            child1.genetic_representation = c1
            child2.genetic_representation = c2
            new_population.append(child1)
            new_population.append(child2)
        self.population = new_population

    def mutation(self, mutation_rate=0.05, mean=0, scale=0.5):
        for agent in self.population:
            for i, p in enumerate(agent.genetic_representation):
                if np.random.rand() < mutation_rate:
                    agent.genetic_representation[i] = np.random.normal(
                        loc=mean, scale=scale
                    )

    def train(
        self, n_generations
    ):  # Training of the agents// DA SISTEMARE DA CAPIRE COME RESETTARE LÉNVIRONEMENT

        for generation in range(n_generations):
            self.env.reset()  # We need to reset the environemnt at the beginning of each generation
            self.create_population()
            for agent in self.population:
                agent.fitness_score = self.env.fitness_score(agent)
            self.proportionate_selection()
            self.crossover()
            self.mutation()
            self.env.reset()
            print(f"Generation {generation} completed")
            print(
                f"Best fitness score: {max([agent.fitness_score for agent in self.population])}"
            )
            print(f"Best agent: {max(self.population, key=lambda x: x.fitness_score)}")
            print("--------------------------------------------------")
        print("Training completed")
        print(f"Best agent: {max(self.population, key=lambda x: x.fitness_score)}")
        print(
            f"Best fitness score: {max([agent.fitness_score for agent in self.population])}"
        )