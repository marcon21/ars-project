import pygame
from pygame.locals import *
from math import pi, degrees, atan2
import numpy as np
from random import randint, random
from env_evolution import EnvEvolution, PygameEvolvedEnviroment
from evolved_agent import EvolvedAgent
import torch
from nn import NN
from parameters import *

# Import parameters from a file if necessary

# Pygame setup
pygame.init()
window = pygame.display.set_mode(GAME_RES, HWACCEL | HWSURFACE | DOUBLEBUF)
clock = pygame.time.Clock()
dt = 0
pygame.display.set_caption(GAME_TITLE)

# Load neural network model
model = NN(INPUT_SIZE, HIDDEN_SIZE, HIDDEN_SIZE2, OUTPUT_SIZE)
model.load_state_dict(torch.load("./saves/best_last_agent.pth"))

delta_x = 300
delta_y = 300
new_x = X_START + np.random.randint(-delta_x, delta_x)
new_y = Y_START + np.random.randint(-delta_y, delta_y)

# Create agent instance
agent = EvolvedAgent(
    x=new_x,
    y=new_y,
    size=AGENT_SIZE,
    move_speed=30,
    n_sensors=INPUT_SIZE,
    max_distance=MAX_DISTANCE,
    color=AGENT_COLOR,
    controller=model,
)


# Create environment instance
env = PygameEvolvedEnviroment(
    agent=agent,
    instants=INSTANTS,
    grid_size=10,
    height=HEIGHT,
    width=WIDTH,
    w1=W1,
    w2=W2,
    w3=W3,
)
# Load walls or landmarks if necessary
env.load_walls(WALLS_TXT)

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

    # Update agent and display environment
    env.move_agent()
    env.show(window)

    for el in env.visited.keys():
        pygame.draw.circle(window, (0, 0, 255), np.array(el) * 10, 2)

    pygame.display.flip()

    # print(env.explored_terrain, framecount)

    dt = clock.tick(FPS) / 1000
