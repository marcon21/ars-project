import pygame
from pygame.locals import *
from math import pi, degrees, atan2
import numpy as np
from random import randint, random, choice
from env_evolution import EnvEvolution, PygameEvolvedEnviroment
from evolved_agent import EvolvedAgent
import torch
from nn import NN
from parameters import *
import os
from actors import Wall


# Import parameters from a file if necessary

# Pygame setup
pygame.init()
window = pygame.display.set_mode(GAME_RES, HWACCEL | HWSURFACE | DOUBLEBUF)
clock = pygame.time.Clock()
pygame.display.set_caption(GAME_TITLE)

dt = 0
grid_size = GRIDSIZE
map_n = 0
COLORS = ["red", "green", "blue", "yellow", "purple", "orange", "pink", "brown"]


def setup(file="./saves/best_last_agent.pth", x=X_START, y=Y_START, c=AGENT_COLOR):
    # Load neural network model
    model = NN(INPUT_SIZE, HIDDEN_SIZE, HIDDEN_SIZE2, OUTPUT_SIZE)
    model.load_state_dict(torch.load(file))

    # Create agent instance
    agent = EvolvedAgent(
        x=y,
        y=x,
        size=AGENT_SIZE,
        move_speed=30,
        n_sensors=INPUT_SIZE,
        max_distance=MAX_DISTANCE,
        color=c,
        controller=model,
    )

    # Create environment instance
    env = PygameEvolvedEnviroment(
        agent=agent,
        instants=INSTANTS,
        height=HEIGHT,
        width=WIDTH,
        w1=W1,
        w2=W2,
        w3=W3,
        grid_size=grid_size,
    )
    # Load walls or landmarks if necessary
    env.load_walls(WALLS_TXT[map_n])

    return env


files = [
    # "./saves/best_last_agent.pth",
    # "./saves/all/bestof/best_gen_10.pth",
    # "./saves/all/bestof/best_gen_15.pth",
    # "./saves/all/bestof/best_gen_20.pth",
    # "./saves/all/bestof/best_gen_25.pth",
    # "./saves/all/bestof/best_gen_30.pth",
    # "./saves/all/bestof/best_gen_40.pth",
    # "./saves/all/bestof/best_gen_50.pth",
]

files = ["saves/best_last_agent.pth"]

gens_to_load = 0
for i in range(0, gens_to_load, 5):
    # check if file exists
    f = f"./saves/all/bestof/best_gen_{i}.pth"
    if os.path.isfile(f):
        files.append(f)

envs = []


def load_envs():
    global envs

    envs = []
    delta_x = 300
    delta_y = 300
    new_x = X_START + np.random.randint(-delta_x, delta_x)
    new_y = Y_START + np.random.randint(-delta_y, delta_y)

    for f in files:
        envs.append(setup(f, new_x, new_y, c=choice(COLORS)))


load_envs()
start, end = None, None
# Main game loop
while True:
    window.fill(SCREEN_COLOR)
    # Handle events
    for event in pygame.event.get():
        if event.type == QUIT:
            pygame.quit()
            quit()
        if event.type == KEYDOWN:
            if event.key == K_ESCAPE:
                pygame.quit()
                quit()
            if event.key == K_SPACE:
                load_envs()
            if event.key == K_m:
                map_n = (map_n + 1) % len(WALLS_TXT)
                load_envs()
            if event.key == K_n:
                start = (
                    pygame.mouse.get_pos()
                    if start is None
                    else (
                        env.add_wall(
                            Wall(
                                start[0],
                                start[1],
                                pygame.mouse.get_pos()[0],
                                pygame.mouse.get_pos()[1],
                            )
                        ),
                        None,
                    )[1]
                )
            if event.key == K_s:
                env.save_walls("./data/walls/walls7.txt")

    # Update agent and display environment
    for i, env in enumerate(envs):
        env.move_agent()
        env.show(window)
        # env.draw_sensors(window)

        if i == 0:
            for el in env.visited.keys():
                pygame.draw.circle(
                    window,
                    (0, 0, 255),
                    np.array(el) * grid_size + [AGENT_SIZE / 2, AGENT_SIZE / 2],
                    2,
                )
            # print(env.collisions)

    pygame.display.flip()

    dt = clock.tick(FPS) / 1000
