import pygame
from pygame.locals import *

pygame.init()


GAME_RES = WIDTH, HEIGHT = 800, 600
FPS = 60
GAME_TITLE = "ARS"

window = pygame.display.set_mode(GAME_RES, HWACCEL | HWSURFACE | DOUBLEBUF)
clock = pygame.time.Clock()
dt = 0
pygame.display.set_caption(GAME_TITLE)

player_pos = pygame.Vector2(window.get_width() / 2, window.get_height() / 2)

running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    window.fill("gray")
    pygame.draw.circle(window, "red", player_pos, 40)

    pygame.display.flip()

    dt = clock.tick(FPS) / 1000

pygame.quit()
quit()
