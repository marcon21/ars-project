from env import Enviroment
from actors import Agent
from nn import NN
from evolved_agent import EvolvedAgent
from env_evolution import EnvEvolution
from utils import create_pairs
import math
import numpy as np
import torch
import random
from parameters import *
from copy import deepcopy


# Author: Aurora Pia Ghiardelli
class Evolution:
    def __init__(
        self,
        initial_population_size,
        input_dim,
        hidden_dim,
        layer_dim,
        output_dim,
        mutation_rate=0.1,
        elitism_rate=0.1,
    ):
        self.initial_population_size = initial_population_size
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.layer_dim = layer_dim
        self.output_dim = output_dim
        self.mutation_rate = mutation_rate
        self.elitism = int(initial_population_size * elitism_rate)

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
        self.sorted_population_with_scores = sorted(
            populations_scores, key=lambda x: x[1], reverse=True
        )

        # Extract the sorted population and fitness scores
        sorted_population, sorted_fitness_scores = zip(
            *self.sorted_population_with_scores
        )

        # Calculate rank-based probabilities
        ranks = np.arange(1, len(sorted_population) + 1)
        probabilities = 1 / ranks
        probabilities /= np.sum(probabilities)

        # Perform selection based on the calculated probabilities
        selected_indices = np.random.choice(
            len(sorted_population),
            size=len(self.population),
            p=probabilities,
            replace=True,
        )
        # Half the population is selected
        # selected_indices = selected_indices[: len(selected_indices) // 2]
        self.population = [sorted_population[i] for i in selected_indices]

    def choose_parents(self):
        # Extract the sorted population and fitness scores
        sorted_population, sorted_fitness_scores = zip(
            *self.sorted_population_with_scores
        )

        fittest = sorted_population[-self.elitism :]
        return random.choice(fittest)

    def mutation(self, genome):
        for k in genome:
            if "weight" in k or "bias" in k:
                noise = torch.randn_like(genome[k]) * self.mutation_rate
                # Gaussian mutation of the genetic representation of the agent
                genome[k] += noise

        return genome

    def crossover(self):
        new_population = []
        for _ in range(len(self.population)):
            parent1 = self.choose_parents()
            parent2 = self.choose_parents()
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
        new_genome = {}

        for k in gen_a:
            alpha = np.random.rand()
            # new_genome[k] = alpha * a[i] + (1 - alpha) * b[i]
            new_genome[k] = gen_a[k] * alpha + gen_b[k] * (1 - alpha)

        new_genome = self.mutation(new_genome)

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
