from parameters import *
import pygame
from pygame.locals import *
from math import pi, degrees, atan2
import numpy as np
from random import randint
from random import random as rand
from env_evolution import EnvEvolution, PygameEvolvedEnviroment
from evolved_agent import EvolvedAgent
import torch
from nn import NN

# Pygame setup
pygame.init()
window = pygame.display.set_mode(GAME_RES, HWACCEL | HWSURFACE | DOUBLEBUF)
clock = pygame.time.Clock()
dt = 0
pygame.display.set_caption(GAME_TITLE)

model = NN(INPUT_SIZE, HIDDEN_SIZE, HIDDEN_SIZE2, OUTPUT_SIZE)
model.load_state_dict(torch.load("./saves/best_last_agent.pth"))
agent = EvolvedAgent(
    x=X_START,
    y=Y_START,
    size=AGENT_SIZE,
    move_speed=30,
    n_sensors=INPUT_SIZE,
    max_distance=MAX_DISTANCE,
    color=AGENT_COLOR,
    controller=model,
)


env = PygameEvolvedEnviroment(agent=agent)
env.load_landmarks(LANDMARK_TXT, LANDMARK_SIZE, LANDMARK_COLOR)
env.load_walls(WALLS_TXT)

while True:
    window.fill(SCREEN_COLOR)

    events = pygame.event.get()
    for event in events:
        if event.type == QUIT:
            quit()
        if event.type == KEYDOWN:
            if event.key == K_ESCAPE:
                quit()

    FPS = 60

    # update window
    env.move_agent()
    env.show(window)

    pygame.display.flip()

    dt = clock.tick(FPS) / 1000
