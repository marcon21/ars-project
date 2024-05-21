from env import Enviroment
from actors import Agent
from nn import NN
from evolved_agent import EvolvedAgent
from env_evolution import PygameEvolvedEnviroment, EnvEvolution
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
            size=len(self.population),
            p=probabilities,
            replace=True,
        )
        self.population = [sorted_population[i] for i in selected_indices]

    def mutation(self, mutation_rate=0.5, mean=0, scale=1):
        for env in self.population:
            for i in range(len(env.agent.genome)):
                if np.random.rand() < mutation_rate:
                    # Applica una mutazione gaussiana al gene
                    env.agent.genome[i] += np.random.normal(loc=mean, scale=scale)

    def crossover(self):

        pairs = create_pairs(self.population, fertility_rate=0.5)
        new_population = []
        i = 0
        for pair in pairs:
            parent1, parent2 = pair
            child1 = EvolvedAgent(
                x=parent1.agent.pos[0],
                y=parent1.agent.pos[1],
                controller=self.create_model(),
                move_speed=5,
                size=40,
                n_sensors=10,
                max_distance=200,
                color="red",
            )

            env1 = PygameEvolvedEnviroment(child1)
            child2 = EvolvedAgent(
                x=parent2.agent.pos[0],
                y=parent2.agent.pos[1],
                controller=self.create_model(),
                move_speed=5,
                size=40,
                n_sensors=10,
                max_distance=200,
                color="red",
            )

            env2 = PygameEvolvedEnviroment(child2)
            gen1 = []
            gen2 = []
            for i, (p1, p2, c1, c2) in enumerate(
                # Crossover between the two parents
                zip(
                    parent1.agent.genome,
                    parent2.agent.genome,
                    child1.genome,
                    child2.genome,
                )
            ):

                alpha = np.random.rand()  # Random number between 0 and 1
                c1 = alpha * p1 + (1 - alpha) * p2

                gen1.append(c1)
                # Linear combination of the two parents' genetic representations
                c2 = alpha * p2 + (1 - alpha) * p1

                gen2.append(c2)
            child1.genome = gen1  # Update the genetic representation of the children
            child2.genome = gen2

            child1.controller.set_weights(
                gen1
            )  # Update the weights of the children's neural networks
            child2.controller.set_weights(gen2)
            new_population.append(env1)
            new_population.append(env2)

        self.population = new_population

    def create_model(self):

        return NN(N_SENSORS, activation=ACTIVATION)

    def create_genetic_representation(self, model):

        representation = []
        for p in model.parameters():
            representation.append(p.data.numpy().flatten())

        return representation

    def create_population(self):
        self.population = []
        # Weights = []
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
            agent.controller.set_weights(agent.controller.get_weights())
            env = PygameEvolvedEnviroment(agent)

            """
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
            """
            
            self.population.append(env)


"""

    def train(
        self, n_generations
    ):  # Training of the agents// DA SISTEMARE DA CAPIRE COME RESETTARE LÉNVIRONEMENT
        self.create_population()

        for generation in range(n_generations):
            
            for agent in self.population:
                agent.env.reset()
                agent.env.
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

"""
