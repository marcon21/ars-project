from parameters import *
import pygame
from pygame.locals import *
from math import pi, degrees, atan2
import numpy as np
from random import randint
from random import random as rand
from env_evolution import EnvEvolution, PygameEvolvedEnviroment
from evolved_agent import EvolvedAgent
import pickle

# Pygame setup
pygame.init()
window = pygame.display.set_mode(GAME_RES, HWACCEL | HWSURFACE | DOUBLEBUF)
clock = pygame.time.Clock()
dt = 0
pygame.display.set_caption(GAME_TITLE)

env = pickle.load(open("./saves/best_agent.pkl", "rb"))
env = PygameEvolvedEnviroment(agent=env.agent, walls=env.walls)
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
