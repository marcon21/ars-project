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
    size=10,
    move_speed=base_move_speed,
    n_sensors=10,
    max_distance=200,
    color="green",
)
mean = np.array([3, 3, 3])
# mean = np.array([window.get_width() / 2, window.get_height() / 2, agent.direction])
cov_matrix = np.diag([20, 20, 20])
R = np.diag([10, 10, 10])
Q = np.diag([1, 1, 1])
env = PygameEnviroment(agent=agent)
env.load_walls("walls.txt")

land1 = Landmark(478, 345, 5, "a", "purple")
land2 = Landmark(899, 345, 5, "b", "purple")
land3 = Landmark(998, 70, 5, "c", "purple")
land4 = Landmark(999, 598, 5, "d", "purple")
land5 = Landmark(996, 598, 5, "e", "purple")
land6 = Landmark(997, 876, 5, "f", "purple")
land7 = Landmark(2, 68, 5, "g", "purple")
land8 = Landmark(999, 69, 5, "h", "purple")
land9 = Landmark(0, 68, 5, "i", "purple")
land10 = Landmark(0, 880, 5, "j", "purple")
land11 = Landmark(0, 878, 5, "k", "purple")
land12 = Landmark(999, 877, 5, "l", "purple")
land13 = Landmark(3, 181, 5, "m", "purple")
land14 = Landmark(577, 181, 5, "n", "purple")
land15 = Landmark(895, 343, 5, "o", "purple")
land16 = Landmark(997, 343, 5, "p", "purple")
land17 = Landmark(481, 344, 5, "q", "purple")
land18 = Landmark(481, 437, 5, "r", "purple")
land19 = Landmark(349, 183, 5, "s", "purple")
land20 = Landmark(349, 602, 5, "t", "purple")
land21 = Landmark(543, 556, 5, "u", "purple")
land22 = Landmark(995, 556, 5, "v", "purple")
land23 = Landmark(207, 302, 5, "w", "purple")
land24 = Landmark(207, 735, 5, "x", "purple")
land25 = Landmark(802, 735, 5, "y", "purple")
land26 = Landmark(802, 650, 5, "z", "purple")
land27 = Landmark(548, 650, 5, "1", "purple")
land28 = Landmark(678, 74, 5, "2", "purple")
land29 = Landmark(678, 280, 5, "3", "purple")
land30 = Landmark(910, 280, 5, "4", "purple")
land31 = Landmark(910, 181, 5, "5", "purple")
land32 = Landmark(100, 361, 5, "6", "purple")
land33 = Landmark(805, 361, 5, "7", "purple")
land34 = Landmark(805, 874, 5, "8", "purple")
land35 = Landmark(85, 331, 5, "9", "purple")
land36 = Landmark(85, 181, 5, "10", "purple")
land37 = Landmark(886, 559, 5, "11", "purple")
land38 = Landmark(786, 559, 5, "12", "purple")
land39 = Landmark(714, 450, 5, "13", "purple")
land40 = Landmark(996, 450, 5, "14", "purple")
land41 = Landmark(712, 451, 5, "15", "purple")
land42 = Landmark(712, 506, 5, "16", "purple")
land43 = Landmark(242, 181, 5, "17", "purple")
land44 = Landmark(206, 181, 5, "18", "purple")

for i in range(1, 45):
    landmark_name = f"land{i}"
    landmark = globals()[landmark_name]
    env.add_landmark(landmark)

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
    env.agent.move_speed = base_move_speed * dt * 0.5 * (not pause_state)

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
