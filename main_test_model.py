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

# Create agent instance
agent = EvolvedAgent(
    x=120,
    y=100,
    size=AGENT_SIZE,
    move_speed=30,
    n_sensors=INPUT_SIZE,
    max_distance=MAX_DISTANCE,
    color=AGENT_COLOR,
    controller=model,
)

# Create environment instance
env = PygameEvolvedEnviroment(agent=agent)
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
    pygame.display.flip()

    dt = clock.tick(FPS) / 1000
