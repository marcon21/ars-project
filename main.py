import pygame
from pygame.locals import *
from actors import Agent, Wall
from env import PygameEnviroment, Enviroment
from utils import intersection, distance_from_wall
from math import pi

pygame.init()


GAME_RES = WIDTH, HEIGHT = 800, 600
FPS = 60
GAME_TITLE = "ARS"

window = pygame.display.set_mode(GAME_RES, HWACCEL | HWSURFACE | DOUBLEBUF)
clock = pygame.time.Clock()
dt = 0
pygame.display.set_caption(GAME_TITLE)

agent = Agent(
    x=window.get_width() / 2,
    y=window.get_height() / 2,
    size=40,
    move_speed=10,
    color="green",
)

env = PygameEnviroment(agent=agent)
# env.add_wall(Wall(100, 100, 200, 100))
env.add_wall(Wall(600, 500, 600, 200))
env.add_wall(Wall(50, 500, 200, 200))

rotation_size = pi / 180 * 10

running = True
while running:
    window.fill("gray")

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        if event.type == KEYDOWN:
            if event.key == K_ESCAPE:
                running = False
            if event.key == K_q:
                agent.rotate(-rotation_size)
            if event.key == K_e:
                agent.rotate(rotation_size)
            if event.key == K_w:
                pass

    env.agent.move_speed = 50 * dt * 1
    env.move_agent()

    # print(env.agent.pos)

    env.draw_sensors(window, n_sensors=20, max_distance=400)
    env.show(window)

    pygame.display.flip()

    dt = clock.tick(FPS) / 1000

pygame.quit()
quit()
