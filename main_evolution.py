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
import time


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
        dt=10,
    )
    evl.create_population()

    print("Size of the population:", len(evl.population))
    print("Number of generations:", GENERATIONS)
    print("Number of instants per simulation:", INSTANTS)

    maps = []
    map_paths = WALLS_TXT
    for path in map_paths:
        # print(path)
        evl.population[0].load_walls(path)
        w = deepcopy(evl.population[0].walls)
        maps.append(w)

    # Simulation for each generation
    try:
        for generation in range(GENERATIONS):
            start_t = time.time()
            # if generation <= 15:
            #     W1 = 0.95
            #     W2 = 0.05
            #     W3 = 0
            # else:
            #     W1 = 0.8
            #     W2 = 0.2
            #     W3 = 0

            print(f"Generation {generation} - Simulating...")

            # random start location
            delta_x = 300
            delta_y = 300
            new_x = X_START + np.random.randint(-delta_x, delta_x)
            new_y = Y_START + np.random.randint(-delta_y, delta_y)
            new_x = X_START
            new_y = Y_START
            map_choice = random.choice(maps)
            for env in evl.population:
                env.agent.x = new_x
                env.agent.y = new_y
                env.walls = deepcopy(map_choice)
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

            end_t = time.time()

            fitness_scores = np.array(
                fitness_scores[:]
            )  # Convert shared array to numpy array

            if np.argmax(fitness_scores) == 0:
                s = "Best agent is the same as last generation"
            else:
                s = "The student has become the master"

            s = f"Generation {generation} - Average Fitness scores: {np.mean(fitness_scores)} \t - Best Fitness score: {np.max(fitness_scores)} \t - {s} \t - Training time: {end_t - start_t} s"
            print(s)

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

            # if generation % 10 == 0 and generation != 0:
            #     INSTANTS *= 2

    # Save best agent
    finally:
        model = best_agent.agent.controller
        torch.save(model.state_dict(), "./saves/best_last_agent.pth")
