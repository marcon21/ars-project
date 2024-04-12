import pygame
from pygame.locals import *
from typing import List
from actors import Agent, Wall
from utils import intersection, distance_from_wall
import numpy as np


class Enviroment:
    def __init__(self, agent: Agent, walls: List[Wall] = []) -> None:
        self.walls = walls
        self.agent = agent

    def add_wall(self, wall: Wall):
        self.walls.append(wall)

    def move_agent(self):
        move_vector = self.agent.direction_vector * self.agent.move_speed
        for wall in self.walls:
            current_d = distance_from_wall(wall, self.agent.pos)

            if current_d <= self.agent.size:
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

        # def rotate_clock():
        #     return np.array([[0, 1], [-1, 0]])

        # top_start = (
        #     self.agent.pos
        #     + self.agent.direction_vector * rotate_clock() * self.agent.size
        # )
        # top_end = (
        #     self.agent.pos
        #     + move_vector
        #     + self.agent.direction_vector * rotate_clock() * self.agent.size
        # )
        # bot_start = (
        #     self.agent.pos
        #     - self.agent.direction_vector * rotate_clock() * self.agent.size
        # )
        # bot_end = (
        #     self.agent.pos
        #     + move_vector
        #     - self.agent.direction_vector * rotate_clock() * self.agent.size
        # )

        self.agent.apply_vector(move_vector)

    def get_sensor_data(self, n_sensors=8, max_distance=200):
        sensor_data = []
        for i in range(n_sensors):
            current_angle = self.agent.direction + i * np.pi / (n_sensors / 2)
            sensor = Wall(
                self.agent.pos[0],
                self.agent.pos[1],
                self.agent.pos[0] + max_distance * np.cos(current_angle),
                self.agent.pos[1] + max_distance * np.sin(current_angle),
            )

            d = max_distance
            int_point = None
            for wall in self.walls:
                intersection_point = intersection(sensor, wall)
                if intersection_point:
                    distance = np.linalg.norm(
                        np.array(intersection_point) - np.array(self.agent.pos)
                    )
                    if distance < d:
                        d = distance
                        int_point = intersection_point

            sensor_data.append((d, int_point))

        return sensor_data


class PygameEnviroment(Enviroment):
    def __init__(self, agent: Agent, walls: List[Wall] = [], color="black"):
        super().__init__(agent, walls=walls)
        pass

    def show(self, window):
        for wall in self.walls:
            pygame.draw.line(window, "black", wall.start, wall.end, width=5)

        agent_color = self.agent.color
        for wall in self.walls:
            dist = distance_from_wall(wall, self.agent.pos)
            if dist == self.agent.size:
                agent_color = "blue"
            if dist < self.agent.size:
                agent_color = "red"

        pygame.draw.circle(window, agent_color, self.agent.pos, self.agent.size)

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

    def draw_sensors(self, window, n_sensors=10, max_distance=200):
        sensor_data = self.get_sensor_data(
            n_sensors=n_sensors, max_distance=max_distance
        )
        for i in range(n_sensors):
            c = "green"
            if sensor_data[i][1] is not None:
                c = "red"
                pygame.draw.circle(window, "red", sensor_data[i][1], 10)

            pygame.draw.line(
                window,
                c,
                self.agent.pos,
                (
                    self.agent.pos[0]
                    + sensor_data[i][0]
                    * np.cos(self.agent.direction + i * np.pi / (n_sensors / 2)),
                    self.agent.pos[1]
                    + sensor_data[i][0]
                    * np.sin(self.agent.direction + i * np.pi / (n_sensors / 2)),
                ),
                width=2,
            )
