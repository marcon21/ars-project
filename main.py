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


# PYGAME SETUP
pygame.init()
GAME_RES = WIDTH, HEIGHT = 1000, 1000
FPS = 60
GAME_TITLE = "ARS"
window = pygame.display.set_mode(GAME_RES, HWACCEL | HWSURFACE | DOUBLEBUF)
clock = pygame.time.Clock()
dt = 0
pygame.display.set_caption(GAME_TITLE)


# AGENT AND ENV SETUP
base_move_speed = 50

agent = Agent(
    x=rand() * window.get_width(),
    y=rand() * window.get_height(),
    size=30,
    move_speed=base_move_speed,
    n_sensors=10,
    max_distance=200,
    color="green",
)
# mean = np.array([agent.pos[0], agent.pos[1], agent.direction])
mean = np.array([window.get_width() / 2, window.get_height() / 2, agent.direction])
cov_matrix = np.diag([0.001, 0.001, 0.001])
R = np.diag([0, 0, 0])
Q = np.diag([0.01, 0.01, 0.01])
env = PygameEnviroment(agent=agent)
# env.load_walls("walls.txt")

land1 = Landmark(100, 100, 50, "a", "purple")
land2 = Landmark(900, 900, 40, "b", "purple")
land3 = Landmark(500, 500, 40, "c", "purple")
land4 = Landmark(200, 250, 40, "c", "purple")
land5 = Landmark(300, 200, 40, "c", "purple")
env.add_landmark(land1)
env.add_landmark(land2)
env.add_landmark(land3)
env.add_landmark(land4)
env.add_landmark(land5)

kfr = PygameKF(env, mean, cov_matrix, R, Q)


def reset_agent():
    agent.pos = (window.get_width() / 2, window.get_height() / 2)
    agent.direction_vector = np.array([1, 0])


rotation_size = pi / 180 * 2
pause_state = False
show_text = False
start = None
end = None


# INSTRUCTIONS
print("\n", "-" * 20, "\n")
print(f"Press 'q' or 'ARROW L' to rotate left by {rotation_size * 180/pi} degrees")
print(f"Press 'e' or 'ARROW R' to rotate right by {rotation_size * 180/pi} degrees")
print("Press 'n' to add a wall")
print("Press 's' to save walls to file")
print("Press 'l' to load walls from file")
print("Press 'BACKSPACE' to remove all walls")
print("Press 'r' to reset agent")
print("Press 't' to toggle text visibility")
print("Press 'SPACE' to toggle movement")
print()


# PYGAME MAIN LOOP
running = True
while running:
    window.fill("gray")

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
    env.agent.move_speed = base_move_speed * dt * 2 * (not pause_state)

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
