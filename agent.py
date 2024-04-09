import pygame
from pygame.locals import *
import numpy as np


class Agent:
    def __init__(
        self, x=0, y=0, direction=0, move_speed=1, size=40, color="red"
    ) -> None:
        self.pos = np.array([x, y])
        self.direction = direction
        self.move_speed = move_speed
        self.size = size
        self.color = color

    def move_forward(self):
        self.pos[0] += self.move_speed * np.cos(self.direction)
        self.pos[1] += self.move_speed * np.sin(self.direction)

    def rotate(self, angle):
        self.direction += angle

    def move(self, angle):
        self.rotate(angle)
        self.move_forward()
