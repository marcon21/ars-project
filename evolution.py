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
        dt=1,
    ):
        self.initial_population_size = initial_population_size
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.layer_dim = layer_dim
        self.output_dim = output_dim
        self.mutation_rate = mutation_rate
        self.elitism = int(initial_population_size * elitism_rate)
        self.dt = dt

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
        populations_scores = list(zip(self.population, fitness_scores))
        sorted_population = sorted(populations_scores, key=lambda x: x[1], reverse=True)

        # Extract the sorted population and fitness scores
        sorted_population, sorted_fitness_scores = zip(*sorted_population)
        self.population = sorted_population[: len(sorted_population) // 8]

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
                noise = np.random.normal(0, self.mutation_rate, genome[k].shape)
                genome[k] += noise

        return genome

    def crossover(self):
        pairs = np.random.choice(
            self.population, size=(self.initial_population_size - 2, 2)
        )

        new_population = [self.population[0], self.population[1]]
        for pair in pairs:
            parent1, parent2 = pair
            new_genome = self.reproduction(parent1.agent.genome, parent2.agent.genome)
            new_genome = self.mutation(new_genome)

            new_env = deepcopy(parent1)
            new_env.agent.controller.set_weights(new_genome)
            new_population.append(new_env)

        self.population = new_population

    def reproduction(self, gen_a, gen_b):
        """
        Reproduction of two agents a and b.
        """
        # Create two children with the same genetic representation as the parents
        assert len(gen_a) == len(gen_b)
        new_genome = deepcopy(gen_a)

        for k in gen_a:
            if "weight" in k or "bias" in k:
                alpha = np.random.rand()
                new_genome[k] = gen_a[k] * alpha + gen_b[k] * (1 - alpha)

        new_genome = self.mutation(new_genome)

        return new_genome

    def create_model(self):
        return NN(
            N_SENSORS,
            activation=ACTIVATION,
            x1=HIDDEN_SIZE,
            x2=HIDDEN_SIZE2,
            dt=self.dt,
        )

    def create_population(self):
        self.population = []
        for i in range(self.initial_population_size):
            nn = self.create_model()
            agent = EvolvedAgent(
                x=X_START,
                y=Y_START,
                n_sensors=N_SENSORS,
                controller=nn,
                size=AGENT_SIZE,
                color=AGENT_COLOR,
                max_distance=MAX_DISTANCE,
            )
            # self.env = EnvEvolution(agent, height=self.env.height, width=self.env.width)
            # agent.controller.set_weights(agent.controller.get_weights())
            env = EnvEvolution(
                agent,
                grid_size=GRIDSIZE,
                height=HEIGHT,
                width=WIDTH,
                w1=W1,
                w2=W2,
                w3=W3,
            )

            self.population.append(env)
