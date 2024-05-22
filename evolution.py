from env import Enviroment
from actors import Agent
from nn import NN
from evolved_agent import EvolvedAgent
from env_evolution import EnvEvolution
from utils import create_pairs
import math
import numpy as np
import torch
from parameters import *


# Author: Aurora Pia Ghiardelli
class Evolution:
    def __init__(
        self,
        initial_population_size,
        input_dim,
        hidden_dim,
        layer_dim,
        output_dim,
    ):
        self.initial_population_size = initial_population_size
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.layer_dim = layer_dim
        self.output_dim = output_dim

    def proportionate_selection(
        self, fitness_scores
    ):  # Selection of the agents based on their fitness score with the proportionate selection method

        sum_fitness = sum(fitness_scores)
        probabilities = [score / sum_fitness for score in fitness_scores]
        self.population = np.random.multinomial(
            len(self.population), probabilities, size=None
        )  # multinomial distribution where each agent is selected with a probability proportional to its fitness score

    def rank_based_selection(self, fitness_scores):
        """
        Selection of the agents based on their fitness score with the rank-based selection method.
        """
        # Combine agents with their fitness scores and sort them
        populations_scores = list(zip(self.population, fitness_scores))
        sorted_population = sorted(populations_scores, key=lambda x: x[1], reverse=True)

        # Extract the sorted population and fitness scores
        sorted_population, sorted_fitness_scores = zip(*sorted_population)

        # Calculate rank-based probabilities
        ranks = np.arange(1, len(sorted_population) + 1)
        probabilities = 1 / ranks
        probabilities /= np.sum(probabilities)

        # Perform selection based on the calculated probabilities
        selected_indices = np.random.choice(
            len(sorted_population),
            size=len(self.population) // 2,
            p=probabilities,
            replace=True,
        )
        # Half the population is selected
        # selected_indices = selected_indices[: len(selected_indices) // 2]
        self.population = [sorted_population[i] for i in selected_indices]

    def mutation(self, mutation_rate=0.5, mean=0, scale=1):
        for env in self.population:
            for i in range(len(env.agent.genome)):
                if np.random.rand() < mutation_rate:
                    # Gaussian mutation of the genetic representation of the agent
                    env.agent.genome[i] += np.random.normal(loc=mean, scale=scale)

    def crossover(self):
        pairs = np.random.choice(
            self.population, size=(self.initial_population_size, 2)
        )

        new_population = []
        for pair in pairs:
            parent1, parent2 = pair
            new_genome = self.reproduction(parent1.agent.genome, parent2.agent.genome)

            agent = EvolvedAgent(
                x=X_START,
                y=Y_START,
                n_sensors=N_SENSORS,
                controller=self.create_model(),
                size=AGENT_SIZE,
                color=AGENT_COLOR,
                max_distance=MAX_DISTANCE,
            )
            agent.controller.set_weights(new_genome)
            env = EnvEvolution(agent)
            new_population.append(env)

        self.population = new_population

    def reproduction(self, gen_a, gen_b):
        """
        Reproduction of two agents a and b.
        """
        # Create two children with the same genetic representation as the parents
        assert len(gen_a) == len(gen_b)
        new_genome = []

        for i in range(len(gen_a)):
            alpha = np.random.rand()
            new_genome.append(alpha * gen_a[i] + (1 - alpha) * gen_b[i])

        return new_genome

    def create_model(self):
        return NN(N_SENSORS, activation=ACTIVATION)

    def create_genetic_representation(self, model):
        representation = []
        for p in model.parameters():
            representation.append(p.data.numpy().flatten())

        return representation

    def create_population(self):
        self.population = []
        for i in range(self.initial_population_size):
            agent = EvolvedAgent(
                x=X_START,
                y=Y_START,
                n_sensors=N_SENSORS,
                controller=self.create_model(),
                size=AGENT_SIZE,
                color=AGENT_COLOR,
                max_distance=MAX_DISTANCE,
            )
            # self.env = EnvEvolution(agent, height=self.env.height, width=self.env.width)
            # agent.controller.set_weights(agent.controller.get_weights())
            env = EnvEvolution(agent)

            self.population.append(env)
