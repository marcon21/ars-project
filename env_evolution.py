from env import Enviroment
from evolved_agent import EvolvedAgent
from actors import Agent, Wall, Landmark
from typing import List
import numpy as np
from copy import deepcopy
from math import degrees, atan2, pi
from utils import distance_from_wall, intersection, angle_from_vector, intersection_line_circle
import torch
from nn import NN
from parameters import *
import matplotlib.pyplot as plt
from numpy import ceil
import pygame
import math

class EnvEvolution(Enviroment):
    def __init__(
        self,
        agent: EvolvedAgent,
        walls: List[Wall] = [],
        landmarks: List[Landmark] = [],
        height=800,
        width=800,
        instants=1000,
        w1=1,
        w2=1,
        w3=0.2,
        
    ):
        super().__init__(agent, walls, landmarks)
        self.height = height
        self.width = width
        self.map = np.zeros((self.width, self.height))
        self.collisions = 0
        self.movements = 0
        self.instants = instants
        self.W1 = w1
        self.W2 = w2
        self.W3 = w3
        self.distance = self.agent.max_distance * np.ones(self.instants)
        self.path = []
        self.map = np.zeros((self.width, self.height))

    def reset(self, random=False):
        if random:
            self.agent.pos = np.array(
                [np.random.randint(0, self.width), np.random.randint(0, self.height)],
                dtype=np.float64,
            )
            self.agent.direction_vector = np.array(
                [np.random.randint(-1, 2), np.random.randint(-1, 2)], dtype=np.float64
            )
        else:
            self.agent.pos = np.array([self.width // 2, self.height // 2], dtype=np.float64)
        self.collisions = 0
        self.movements = 0
        self.map = np.zeros((self.width, self.height))
        self.distance = self.agent.max_distance * np.ones(self.instants)
        self.path = []
        self.ang = []

    def move_agent(self, dt=100):
        try:
            distances = np.array([data[1][0] for data in  self.get_sensor_data(self.agent.n_sensors, self.agent.max_distance)], dtype=np.float32)
            vl, vr = self.agent.controller.forward(torch.tensor(distances, dtype=torch.float))
        except Exception as e:
            print(e)
            vl, vr = 0, 0
        v, w = (vl + vr) / 2, (vr - vl) / (self.agent.size * 2)
        self.ang.append(w)
        dx,dy, dtheta = 0,0, w * dt
        if w == 0:
            dx = v * dt * np.cos(self.agent.direction)
            dy = v * dt * np.sin(self.agent.direction)
        else:
            R = v / w
            dx = R * (np.sin(self.agent.direction + dtheta) - np.sin(self.agent.direction))
            dy = -R * (np.cos(self.agent.direction + dtheta) - np.cos(self.agent.direction))

        dx = np.clip(dx, -10, 10)
        dy = np.clip(dy, -10, 10)
        move_vector = np.array([dx, dy])
        if move_vector[0] == 0 and move_vector[1] == 0:
            return
        for wall in self.walls:
            current_d = distance_from_wall(wall, self.agent.pos)

            if current_d <= self.agent.size:
                # Vector of the wall direction
                self.collisions += 1
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
                #print("ILLEGAL MOVE")
                return
            
            
        #print(f"dx: {dx}, dy: {dy}, dtheta: {dtheta}")
            
        self.agent.apply_vector(move_vector)
        # Set visited positions to 1 within -10 to +10 range around agent's position
        x_start = max(0, int(self.agent.pos[0]) - 10)
        x_end = min(self.map.shape[0], int(self.agent.pos[0]) + 10 + 1)
        y_start = max(0, int(self.agent.pos[1]) - 10)
        y_end = min(self.map.shape[1], int(self.agent.pos[1]) + 10 + 1)

        # Efficiently set the range using numpy slicing
        self.map[x_start:x_end, y_start:y_end] = 1

        # Append the new position to the path
        self.path.append((self.agent.pos[0], self.agent.pos[1]))

        # Rotate the agent
        self.agent.rotate(np.deg2rad(dtheta))

    def fitness_score(self) -> float:
        
        return (
            self.explored_terrain * 100 - sum(self.ang) / len(self.ang)
            #np.mean(1 / self.distance)
            #+ np.exp(-self.collisions)
        )
    @property
    def explored_terrain(self) -> float:
        return np.sum(self.map) / (self.width * self.height)

    def visualize_movement(self):
        if self.path == []:
            return
        plt.figure(figsize=(10, 10))
        plt.imshow(self.map.T, origin='lower', cmap='gray_r')
        plt.colorbar(label='Explored')
        plt.scatter(self.agent.pos[0], self.agent.pos[1], color='green', s=100, label='Agent')
        for wall in self.walls:
            plt.plot([wall.start[0], wall.end[0]], [wall.start[1], wall.end[1]], 'b-', linewidth=2)
        path = np.array(self.path)
        plt.plot(path[:, 0], path[:, 1], 'r-', label='Path')
        plt.title('Agent Movement and Explored Terrain')
        plt.xlabel('Width')
        plt.ylabel('Height')
        plt.legend()
        plt.show()


class PygameEvolvedEnviroment(EnvEvolution):

    def __init__(
        self,
        agent,
        walls: List[Wall] = [],
        color="black",
        landmarks: List[Landmark] = [],
        height=800,
        width=800,
        instants=1000,
        w1=0.5,
        w2=0.3,
        w3=0.2,
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
