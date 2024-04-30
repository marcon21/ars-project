import pygame
from pygame.locals import *
import numpy as np


class Wall:
    def __init__(self, x1, y1, x2, y2):
        self.start = (x1, y1)
        self.end = (x2, y2)


class Landmark:
    def __init__(self, x, y, r):
        self.cord = (x, y)
        self.ray = r


class Agent:
    def __init__(self, x=0, y=0, move_speed=5, size=40, color="red") -> None:
        self.pos = np.array([x, y])
        self.size = size
        self.color = color
        self.direction_vector = np.array([1, 0])
        self.move_speed = move_speed

    @property
    def direction(self):
        return np.arctan2(self.direction_vector[1], self.direction_vector[0])

    def rotate(self, angle):
        self.direction_vector = np.dot(
            np.array(
                [
                    [np.cos(angle), -np.sin(angle)],
                    [np.sin(angle), np.cos(angle)],
                ]
            ),
            self.direction_vector,
        )

    def move(self):
        self.apply_vector(self.direction_vector * self.move_speed)

    def apply_vector(self, vector):
        self.pos += vector
