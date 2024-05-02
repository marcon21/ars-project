import pygame
from pygame.locals import *
from actors import Agent, Wall, Landmark
from env import PygameEnviroment, Enviroment
from kalman_filter import Kalman_Filter, PygameKF
from utils import intersection, distance_from_wall
from math import pi
import numpy as np
from random import randint
from random import random as rand
from map import *

def reset_agent():
    agent.pos = (window.get_width() / 2, window.get_height() / 2)
    agent.direction_vector = np.array([1, 0])

# Pygame setup
pygame.init()
window = pygame.display.set_mode(GAME_RES, HWACCEL | HWSURFACE | DOUBLEBUF)
clock = pygame.time.Clock()
dt = 0
pygame.display.set_caption(GAME_TITLE)

# Initialize agent
agent = Agent(x=rand() * WIDTH, y=rand() * HEIGHT, size=AGENT_SIZE,
              move_speed=BASE_MOVE_SPEED, n_sensors=SENSORS, max_distance=RANGE, color=AGENT_COLOR)

# Initialize environment and load landmarks
env = PygameEnviroment(agent=agent)
env.load_landmarks(LANDMARK_TXT, LANDMARK_SIZE, LANDMARK_COLOR)

# Initialize Kalman Filter
kfr = PygameKF(env, MEAN, COV_MATRIX, R, Q)

# Set initial state
pause_state, show_text = False, False
start, end = None, None

print(INSTRUCTIONS)


while True:
    window.fill(SCREEN_COLOR)
    for event in pygame.event.get():
        if event.type == QUIT: quit()
        if event.type == KEYDOWN:
            if event.key == K_ESCAPE: quit()
            if event.key in (K_q, K_LEFT): agent.turn_direction -= ROTATION_SIZE
            if event.key in (K_e, K_RIGHT): agent.turn_direction += ROTATION_SIZE
            if event.key == K_n: start = pygame.mouse.get_pos() if start is None else (env.add_wall(Wall(start[0], start[1], pygame.mouse.get_pos()[0], pygame.mouse.get_pos()[1])), None)[1]
            if event.key == K_s: env.save_walls("walls.txt")
            if event.key == K_l: env.load_walls("walls.txt"), reset_agent()
            if event.key == K_BACKSPACE: env.walls.clear()
            if event.key == K_r: reset_agent()
            if event.key == K_t: show_text = not show_text
            if event.key == K_SPACE: pause_state = not pause_state
    if start: pygame.draw.line(window, "blue", start, pygame.mouse.get_pos(), 5)
    env.agent.move_speed = BASE_MOVE_SPEED * dt * 0.5 * (not pause_state)
    if not pause_state: env.move_agent()
    env.draw_sensors(window, show_text=show_text), env.show(window), kfr.correction(), kfr.show(window), pygame.display.flip()
    dt = clock.tick(FPS) / 1000
