from os import environ

environ["PYGAME_HIDE_SUPPORT_PROMPT"] = "1"

from parameters import *
from evolution import Evolution
import multiprocessing as mp
import numpy as np
import pygame
from pygame.locals import *
from tqdm import tqdm
import pickle
import random
from copy import deepcopy
import pandas as pd
from sklearn.metrics.pairwise import cosine_distances
from scipy.spatial.distance import pdist
import matplotlib.pyplot as plt


def run_simulation(env, i, fitness_scores):
    env.reset()
    for _ in range(INSTANTS):
        env.move_agent()

    fitness_scores[i] = env.fitness_score()


if __name__ == "__main__":
    # Initialize Evolution
    import torch
    import os

    if os.name == "nt":
        mp.set_start_method("spawn")
    else:
        mp.set_start_method("fork")

    evl = Evolution(
        initial_population_size=AGENT_NUMBER,
        input_dim=INPUT_SIZE + 4,
        hidden_dim=32,
        layer_dim=4,
        output_dim=2,
        mutation_rate=0.1,
        elitism_rate=0.1,
        dt=1,
    )
    evl.create_population()

    print("Size of the population:", len(evl.population))
    print("Number of generations:", GENERATIONS)
    print("Number of instants per simulation:", INSTANTS)

    maps = []
    map_paths = [WALLS_TXT]
    for path in map_paths:
        evl.population[0].load_walls(path)
        w = deepcopy(evl.population[0].walls)
        maps.append(w)
    cosine_distances_evolution = []
    pairwise_distances_evolution = []

    # Simulation for each generation
    try:
        for generation in range(GENERATIONS):
            
            genomes = [(env.agent.controller.get_parameters_as_array()) for env in evl.population]
            genomes = pd.DataFrame(genomes)
            cos_distances = cosine_distances(genomes)
            pair_dist = pdist(genomes)
            average_cos_distance = np.mean(cos_distances)
            cosine_distances_evolution.append(average_cos_distance)
            average_pair_distance = np.mean(pair_dist)
            pairwise_distances_evolution.append(average_pair_distance)
            print(f"Generation {generation} - Average Cosine Distance: {average_cos_distance} - Average Pairwise Distance: {average_pair_distance}")
            
        
            
            
        
            print(f"Generation {generation} - Simulating...")

            # random start location
            delta_x = 300
            delta_y = 300
            new_x = X_START + np.random.randint(-delta_x, delta_x)
            new_y = Y_START + np.random.randint(-delta_y, delta_y)
            new_x = X_START
            new_y = Y_START
            for env in evl.population:
                env.agent.x = new_x
                env.agent.y = new_y
                env.walls = deepcopy(random.choice(maps))
                env.w1 = W1
                env.w2 = W2
                env.w3 = W3

            # Using shared array for multiprocessing
            fitness_scores = mp.Array("d", AGENT_NUMBER)
            processes = []
            for i, env in enumerate(evl.population):
                p = mp.Process(target=run_simulation, args=(env, i, fitness_scores))
                processes.append(p)

            for p in processes:
                p.start()

            for p in processes:
                p.join()

            fitness_scores = np.array(
                fitness_scores[:]
            )  # Convert shared array to numpy array

            if np.argmax(fitness_scores) == 0:
                s = "Best agent is the same as last generation"
            else:
                s = "The student has become the master"

            print(
                f"Generation {generation} - Average Fitness scores: {np.mean(fitness_scores)} - Best Fitness score: {np.max(fitness_scores)} - {s}"
            )
            
            

            # Check if the best agent is the same as last generation
            # if np.argmax(fitness_scores) == 0:
            #     if evl.mutation_rate > 0.3:
            #         INSTANTS *= 2
            #         if INSTANTS > 8000:
            #             INSTANTS = 8000
            #         evl.mutation_rate = 0.1
            #         print("INSTANTS increased")
            #     else:
            #         evl.mutation_rate += 0.05
            #         print("Mutation rate increased")
            # else:
            #     evl.mutation_rate = 0.1

            # Save best agent
            best_agent = evl.population[np.argmax(fitness_scores)]
            model = best_agent.agent.controller
            torch.save(
                model.state_dict(),
                f"./saves/all/bestof/best_gen_{generation}.pth",
            )

            # check if folder exists
            if not os.path.exists(f"./saves/all/gen_{generation}"):
                os.makedirs(f"./saves/all/gen_{generation}")
            else:
                # remove all files in folder
                files = os.listdir(f"./saves/all/gen_{generation}")
                for file in files:
                    os.remove(f"./saves/all/gen_{generation}/{file}")

            # Save all agents
            for i, env in enumerate(evl.population):
                model = env.agent.controller
                torch.save(
                    model.state_dict(),
                    f"./saves/all/gen_{generation}/agent_{i}.pth",
                )

            # Evolution steps
            evl.rank_based_selection(fitness_scores)
            evl.crossover()

            if generation % 10 == 0 and generation != 0:
                INSTANTS *= 2

    # Save best agent
    finally:
        model = best_agent.agent.controller
        torch.save(model.state_dict(), "./saves/best_last_agent.pth")
        # Genera l'asse x (numero di generazioni)
        generations = range(1, len(cosine_distances_evolution) + 1)

        plt.figure(figsize=(10, 5))

        plt.subplot(2, 1, 1)
        plt.scatter(generations, cosine_distances_evolution, label='Cosine Distances', color='blue')
        plt.xlabel('Generation')
        plt.ylabel('Distance')
        plt.title('Cosine Distances Over Generations')
        plt.grid(True)
        plt.legend()

        plt.subplot(2, 1, 2)
        plt.scatter(generations, pairwise_distances_evolution, label='Pairwise Distances', color='red')
        plt.xlabel('Generation')
        plt.ylabel('Distance')
        plt.title('Pairwise Distances Over Generations')
        plt.grid(True)
        plt.legend()

        plt.tight_layout()

        plt.savefig('distances_evolution_plot.png')

        plt.show()
                
