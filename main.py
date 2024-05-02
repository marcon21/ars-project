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


# PYGAME SETUP
pygame.init()
window = pygame.display.set_mode(GAME_RES, HWACCEL | HWSURFACE | DOUBLEBUF)
clock = pygame.time.Clock()
dt = 0
pygame.display.set_caption(GAME_TITLE)

agent = Agent(x=rand() * WIDTH, y=rand() * HEIGHT,size=AGENT_SIZE,
            move_speed=BASE_MOVE_SPEED,n_sensors=SENSORS,max_distance=RANGE,color=AGENT_COLOR,)

# mean = np.array([window.get_width() / 2, window.get_height() / 2, agent.direction])

env = PygameEnviroment(agent=agent)
env.load_walls(WALLS_TXT)
env.load_landmarks(LANDMARK_TXT,LANDMARK_SIZE,LANDMARK_COLOR)

kfr = PygameKF(env, MEAN, COV_MATRIX, R, Q)


def reset_agent():
    agent.pos = (window.get_width() / 2, window.get_height() / 2)
    agent.direction_vector = np.array([1, 0])


rotation_size = pi / 180 * 2
pause_state,show_text = False, False
start,end = None,None

print(INSTRUCTIONS)


# PYGAME MAIN LOOP
running = True
while running:
    window.fill(SCREEN_COLOR)

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        if event.type == KEYDOWN:
            if event.key == K_ESCAPE:
                running = False
            if event.key == K_q or event.key == K_LEFT:
                # Rotate left
                # agent.rotate(-rotation_size)
                agent.turn_direction -= rotation_size
            if event.key == K_e or event.key == K_RIGHT:
                # Rotate right
                # agent.rotate(rotation_size)
                agent.turn_direction += rotation_size
            if event.key == K_n:
                # Add wall to the environment
                if start is None:
                    start = pygame.mouse.get_pos()
                else:
                    end = pygame.mouse.get_pos()
                    env.add_wall(Wall(start[0], start[1], end[0], end[1]))
                    start = None
                    end = None
            if event.key == K_s:
                # Save walls
                env.save_walls("walls.txt")
            if event.key == K_l:
                # Load walls
                filename = "walls.txt"
                env.load_walls(filename)
                reset_agent()
            if event.key == K_BACKSPACE:
                # Remove all walls
                env.walls = []
            if event.key == K_r:
                # Reset agent
                reset_agent()
            if event.key == K_t:
                # Show text
                show_text = not show_text
            if event.key == K_SPACE:
                # Toggle movement
                pause_state = not pause_state

    # Draw wall being added
    if start is not None:
        pygame.draw.line(window, "blue", start, (pygame.mouse.get_pos()), 5)

    # Change move speed based on last frame processing time
    env.agent.move_speed = BASE_MOVE_SPEED * dt * 0.5 * (not pause_state)

    # Take step in the phisic simulation and show the environment

    if not pause_state:
        env.move_agent()

    env.draw_sensors(window, show_text=show_text)
    env.show(window)

    kfr.correction()
    kfr.show(window)

    # Update the display
    pygame.display.flip()

    # Calculate the time passed since last frame in ms
    dt = clock.tick(FPS) / 1000

pygame.quit()
quit()
