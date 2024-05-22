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


def run_simulation(env, i, fitness_scores):
    env.reset()
    for _ in range(INSTANTS):
        env.move_agent()
    fitness_scores[i] = env.fitness_score()


if __name__ == "__main__":
    # Initialize Evolution

    mp.set_start_method("spawn")

    multiprocessing = True

    evl = Evolution(
        initial_population_size=AGENT_NUMBER,
        input_dim=INPUT_SIZE + 4,
        hidden_dim=32,
        layer_dim=4,
        output_dim=2,
        mutation_rate=0.1,
        elitism_rate=0.1,
    )
    evl.create_population()

    print("Initializing evolution... multiprocessing:", multiprocessing)
    print("Size of the population:", len(evl.population))
    print("Number of generations:", GENERATIONS)
    print("Number of instants per simulation:", INSTANTS)

    # Load environment configurations
    for env in evl.population:
        env.load_landmarks(LANDMARK_TXT, LANDMARK_SIZE, LANDMARK_COLOR)
        env.load_walls(WALLS_TXT)

    # Simulation for each generation

    for generation in range(GENERATIONS):
        print(f"Generation {generation} - Simulating...")

        # Using shared array for multiprocessing
        if multiprocessing:
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

        else:
            fitness_scores = np.zeros(AGENT_NUMBER)
            for i, env in enumerate(evl.population):
                run_simulation(env, i, fitness_scores)

        print(
            f"Generation {generation} - Average Fitness scores: {np.mean(fitness_scores)}"
        )
        if generation == 0:
            with open("./saves/average_fitness_scores.txt", "w") as f:
                f.write(
                    f"Generation: {generation} ~ Average Fitness: {np.mean(fitness_scores)} ~ std: {np.std(fitness_scores)} \n"
                )
            with open("./saves/fitness_scores.txt", "w") as f:
                f.write(f"Generation: {generation} ~ Fitness: {fitness_scores}\n")

        else:
            with open("./saves/average_fitness_scores.txt", "a") as f:
                f.write(
                    f"Generation: {generation} ~ Fitness: {np.mean(fitness_scores)}~ std: {np.std(fitness_scores)}\n"
                )

            with open("./saves/fitness_scores.txt", "a") as f:
                f.write(f"Generation: {generation} ~ Fitness: {fitness_scores}\n")

        # Evolution steps
        evl.rank_based_selection(fitness_scores)
        evl.crossover()
        # evl.mutation()

    # Save best agent
    best_agent = evl.population[np.argmax(fitness_scores)]
    with open("./saves/best_agent.pkl", "wb") as f:
        pickle.dump(best_agent, f)
