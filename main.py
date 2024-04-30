import pygame
from pygame.locals import *
from actors import Agent, Wall
from env import PygameEnviroment, Enviroment
from utils import intersection, distance_from_wall
from math import pi
import numpy as np


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
    x=window.get_width() / 2,
    y=window.get_height() / 2,
    size=10,
    move_speed=base_move_speed,
    color="green",
)
env = PygameEnviroment(agent=agent)
env.load_walls("walls.txt")
env.load_landmarks("landmarks.txt")
print(env.map())


def reset_agent():
    agent.pos = (window.get_width() / 2, window.get_height() / 2)
    agent.direction_vector = np.array([1, 0])


rotation_size = pi / 180 * 10
move_modifier = 1
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
                agent.rotate(-rotation_size)
            if event.key == K_e or event.key == K_RIGHT:
                # Rotate right
                agent.rotate(rotation_size)
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
                if move_modifier:
                    move_modifier = 0
                else:
                    move_modifier = 1

    # Draw wall being added
    if start is not None:
        pygame.draw.line(window, "blue", start, (pygame.mouse.get_pos()), 5)

    # Change move speed based on last frame processing time
    env.agent.move_speed = base_move_speed * dt * move_modifier * 1

    # Take step in the phisic simulation and show the environment
    env.move_agent()
    env.draw_sensors(window, n_sensors=20, max_distance=400, show_text=show_text)
    env.show(window)
    # env.draw_wall_coordinates(window)
    # Update the display
    pygame.display.flip()

    # Calculate the time passed since last frame in ms
    dt = clock.tick(FPS) / 1000

pygame.quit()
quit()
