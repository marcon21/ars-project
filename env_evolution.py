from env import Enviroment
from evolved_agent import EvolvedAgent
from actors import Agent, Wall, Landmark
from typing import List
import numpy as np
from copy import deepcopy
from math import degrees, atan2, pi
from utils import (
    distance_from_wall,
    intersection,
    angle_from_vector,
    intersection_line_circle,
)
import torch
from nn import NN
from parameters import *
import matplotlib.pyplot as plt
from numpy import ceil
import pygame
import math
import matplotlib.animation as animation

CLIP = 50


class EnvEvolution(Enviroment):

    def __init__(
        self,
        agent: EvolvedAgent,
        walls: List[Wall] = [],
        landmarks: List[Landmark] = [],
        instants=1000,
        w1=0.8,
        w2=0.1,
        w3=0.1,
        grid_size=100,
        height=1000,
        width=1000,
    ):
        super().__init__(agent, walls, landmarks)
        self.collisions = 0
        self.movements = 0
        self.instants = instants
        self.w1 = w1
        self.w2 = w2
        self.w3 = w3
        self.distance = self.agent.max_distance * np.ones(self.instants)
        self.w = []
        self.grid_size = grid_size
        self.visited = {}
        self.total_cells = (width // grid_size) * (height // grid_size)
        self.average_turn = 0
        self.ticks = 0

    def reset(self, random=False):
        self.collisions = 0
        self.movements = 0
        self.distance = self.agent.max_distance * np.ones(self.instants)
        self.visited = {}

    def move_agent(self, dt=1):
        distances = np.array(
            [
                data[1][0]
                for data in self.get_sensor_data(
                    self.agent.n_sensors, self.agent.max_distance, noise=True
                )
            ],
            dtype=np.float32,
        )

        distances = distances / self.agent.max_distance
        distances = torch.tensor(distances, dtype=torch.float)

        vl, vr = self.agent.controller.forward(distances)

        move_vector = self.agent.direction_vector * 5
        theta = (vr - vl) * (pi / 180) * 90

        for wall in self.walls:
            current_d = distance_from_wall(wall, self.agent.pos)

            if current_d <= self.agent.size:
                self.collisions += 1

                # Vector of the wall direction
                wall_vector = np.array(
                    [wall.end[0] - wall.start[0], wall.end[1] - wall.start[1]]
                )
                wall_vector = wall_vector / np.linalg.norm(wall_vector)

                # Vector of the agent parallel to the wall
                parallel_component = np.dot(wall_vector, move_vector) * wall_vector

                # Vector of the agent perpendicular to the wall
                wall_to_agent = self.agent.pos - np.array(
                    distance_from_wall(wall, self.agent.pos, coords=True)
                )
                wall_to_agent = wall_to_agent / np.linalg.norm(wall_to_agent)

                # If the agent is inside the wall push it out
                self.agent.apply_vector(wall_to_agent * (self.agent.size - current_d))
                # Check if the agent is moving towards the wall
                if np.dot(self.agent.direction_vector, -wall_to_agent) > 0:
                    # If the agent is moving towards the wall only consider the parallel component
                    move_vector = parallel_component

        # Check if the agent is making an illegal move
        for wall in self.walls:
            intersection_point = intersection(
                Wall(
                    self.agent.pos[0],
                    self.agent.pos[1],
                    self.agent.pos[0] + move_vector[0],
                    self.agent.pos[1] + move_vector[1],
                ),
                wall,
            )
            if intersection_point:
                # print("ILLEGAL MOVE")
                return

        self.agent.apply_vector(move_vector)
        self.agent.rotate(theta)

        x_cell = int(self.agent.pos[0] // self.grid_size)
        y_cell = int(self.agent.pos[1] // self.grid_size)

        self.visited[(x_cell, y_cell)] = 1

    def fitness_score(self) -> float:
        return (
            self.explored_terrain * self.w1
            + np.exp(-self.collisions / 25) * self.w2
            # + self.average_turn * self.w3
        )

    @property
    def explored_terrain(self) -> float:
        return len(self.visited) / self.total_cells

    def visualize_movement(self):
        if not self.path:
            return

        fig, ax = plt.subplots(figsize=(10, 10))
        ax.imshow(self.map.T, origin="lower", cmap="gray_r")

        # Correct the point size for the agent
        ax.scatter(
            self.agent.pos[0],
            self.agent.pos[1],
            color="green",
            s=(self.agent.size * 2) ** 2,
            label="Agent",
        )

        for wall in self.walls:
            ax.plot(
                [wall.start[0], wall.end[0]],
                [wall.start[1], wall.end[1]],
                "b-",
                linewidth=2,
            )

        # Plot the path
        path = np.array(self.path)
        ax.plot(path[:, 0], path[:, 1], "r-", label="Path")

        ax.set_title("Agent Movement and Explored Terrain")
        ax.set_xlabel("Width")
        ax.set_ylabel("Height")
        ax.legend()
        plt.show()


class PygameEvolvedEnviroment(EnvEvolution):

    def __init__(
        self,
        agent,
        walls: List[Wall] = [],
        color="black",
        landmarks: List[Landmark] = [],
        instants=1000,
        w1=0.5,
        w2=0.3,
        w3=0.2,
        grid_size=10,
        height=1000,
        width=1000,
    ):
        super().__init__(
            agent,
            walls=walls,
            landmarks=[],
            height=height,
            width=width,
            instants=instants,
            w1=w1,
            w2=w2,
            w3=w3,
            grid_size=grid_size,
        )

    def show(self, window):
        for wall in self.walls:
            pygame.draw.line(window, "black", wall.start, wall.end, width=5)

        agent_color = self.agent.color
        # Draw agent
        pygame.draw.circle(window, agent_color, self.agent.pos, self.agent.size)

        # Draw agent orientation
        pygame.draw.line(
            window,
            "orange",
            (self.agent.pos[0], self.agent.pos[1]),
            (
                self.agent.pos[0]
                + 100 * math.cos(angle_from_vector(self.agent.direction_vector)),
                self.agent.pos[1]
                + 100 * math.sin(angle_from_vector(self.agent.direction_vector)),
            ),
            2,
        )

        # pygame.draw.lines(window, "black", False, self.agent.path, 2)
        # self.agent.path = self.agent.path[-1000:]

        # Draw agent direction
        pygame.draw.line(
            window,
            "black",
            self.agent.pos,
            (
                self.agent.pos[0] + self.agent.size * np.cos(self.agent.direction),
                self.agent.pos[1] + self.agent.size * np.sin(self.agent.direction),
            ),
            width=4,
        )

        # Draw Estimated Path based on the agent direction
        path = []
        # temp_agent = deepcopy(self.agent)
        # pygame.draw.lines(window, "blue", False, path, width=2)

        # Draw landmarks
        for landmark in self.landmarks:
            pygame.draw.circle(window, landmark.color, landmark.pos, landmark.size)
            # draw landmark positions
            window.blit(
                pygame.font.Font(None, 15).render(
                    f"({landmark.pos[0]}, {landmark.pos[1]}), {landmark.signature}",
                    True,
                    "green",
                ),
                (landmark.pos[0], landmark.pos[1]),
            )

    def draw_sensors(self, window, show_text=False):
        sensor_data = self.get_sensor_data(
            n_sensors=self.agent.n_sensors, max_distance=self.agent.max_distance
        )

        for i in range(self.agent.n_sensors):
            c = "green"
            if sensor_data[i][0] is not None:
                c = "green"
                pygame.draw.circle(window, c, sensor_data[i][0], 5)

                pygame.draw.line(
                    window,
                    c,
                    self.agent.pos,
                    (
                        self.agent.pos[0]
                        + sensor_data[i][1][0]
                        * np.cos(
                            self.agent.direction
                            + i * np.pi / (self.agent.n_sensors / 2)
                        ),
                        self.agent.pos[1]
                        + sensor_data[i][1][0]
                        * np.sin(
                            self.agent.direction
                            + i * np.pi / (self.agent.n_sensors / 2)
                        ),
                    ),
                    width=2,
                )
