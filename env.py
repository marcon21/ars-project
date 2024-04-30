import pygame
from pygame.locals import *
from typing import List
from actors import Agent, Wall, Landmark
from utils import intersection, distance_from_wall, circle_line_intersection
import numpy as np
import uuid
import math


class Enviroment:
    def __init__(
        self, agent: Agent, walls: List[Wall] = [], landmarks: List[Landmark] = []
    ) -> None:
        self.walls = walls
        self.agent = agent
        self.landmarks = landmarks

    def add_wall(self, wall: Wall):
        self.walls.append(wall)

    def add_landmark(self, landmark: Landmark):
        self.landmarks.append(landmark)

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
                print("ILLEGAL MOVE")
                return

        self.agent.apply_vector(move_vector)

    def get_sensor_data(self, n_sensors=8, max_distance=200):
        sensor_data = []
        sensor_data_landmarks = []
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
            int_point_l = None
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
            for landmark in self.landmarks:

                intersection_point_l = circle_line_intersection(
                    (landmark.cord[0], landmark.cord[1]),
                    landmark.ray,
                    sensor.start,
                    sensor.end,
                )

                if intersection_point_l:

                    distance_l = np.linalg.norm(
                        np.array(intersection_point_l) - np.array(self.agent.pos)
                    )

                    if distance_l < d:
                        print(distance_l, intersection_point_l)
                        d = distance_l
                        int_point_l = intersection_point_l

                    orientation = (
                        math.atan2(
                            landmark.cord[1] - self.agent.pos[1],
                            landmark.cord[0] - self.agent.pos[0],
                        )
                        - current_angle
                    )

            sensor_data_landmarks.append((d, int_point_l))
        return sensor_data, sensor_data_landmarks

    def save_walls(self, filename):
        with open(filename, "w") as f:
            for wall in self.walls:
                f.write(
                    f"{wall.start[0]} {wall.start[1]} {wall.end[0]} {wall.end[1]}\n"
                )

        print("Walls saved to", filename)

    def load_walls(self, filename):
        self.walls = []
        with open(filename, "r") as f:
            for line in f:
                wall = line.split()
                self.add_wall(
                    Wall(int(wall[0]), int(wall[1]), int(wall[2]), int(wall[3]))
                )

        print("Walls loaded from", filename)

    def save_landmarks(self, filename):
        with open(filename, "w") as f:
            for landmark in self.landmarks:
                f.write(f"{landmark.cord[0]} {landmark.cord[1]} {landmark.ray[1]}\n")

        print("Landmark saved to", filename)

    def load_landmarks(self, filename):
        self.landmarks = []
        with open(filename, "r") as f:
            for line in f:
                landmark = line.split()
                print(landmark)
                self.add_landmark(
                    Landmark(int(landmark[0]), int(landmark[1]), int(landmark[2]))
                )

        print("Landmark loaded from", filename)

    def map(self):
        self.coordinates = []
        for wall in self.walls:
            segment_points = bresenham_line(
                wall.start[0], wall.start[1], wall.end[0], wall.end[1]
            )
            for point in segment_points:
                self.coordinates.append((point, 0))

    def density_estimation(self, map, previous_location, control, measurement):

        return


def bresenham_line(x0, y0, x1, y1):
    points = []
    dx = abs(x1 - x0)
    dy = abs(y1 - y0)
    sx = -1 if x0 > x1 else 1
    sy = -1 if y0 > y1 else 1
    err = dx - dy

    while x0 != x1 or y0 != y1:
        points.append((x0, y0))
        e2 = 2 * err
        if e2 > -dy:
            err -= dy
            x0 += sx
        if e2 < dx:
            err += dx
            y0 += sy

    points.append((x1, y1))
    return points


class PygameEnviroment(Enviroment):
    def __init__(
        self,
        agent: Agent,
        walls: List[Wall] = [],
        landmarks: List[Landmark] = [],
        color="black",
    ):
        super().__init__(agent, walls=walls, landmarks=landmarks)
        pass

    def show(self, window):
        for wall in self.walls:
            pygame.draw.line(window, "black", wall.start, wall.end, width=5)
        for landmark in self.landmarks:
            pygame.draw.circle(
                window, "red", (landmark.cord[0], landmark.cord[0]), landmark.ray
            )

        agent_color = self.agent.color
        # for wall in self.walls:
        #     dist = distance_from_wall(wall, self.agent.pos)
        #     if dist == self.agent.size:
        #         agent_color = "blue"
        #     if dist < self.agent.size:
        #         agent_color = "red"

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

    def draw_sensors(self, window, n_sensors=10, max_distance=200, show_text=False):
        sensor_data, sensor_data_landmarks = self.get_sensor_data(
            n_sensors=n_sensors, max_distance=max_distance
        )
        for i in range(n_sensors):
            c = "green"
            """
            if sensor_data[i][1] is not None:
                c = "red"
                pygame.draw.circle(window, "red", sensor_data[i][1], 3)
            """
            if sensor_data_landmarks[i][1] is not None:
                c = "red"
                pygame.draw.circle(window, "red", sensor_data_landmarks[i][1], 3)
            pygame.draw.line(
                window,
                c,
                self.agent.pos,
                (
                    self.agent.pos[0]
                    + sensor_data_landmarks[i][0]
                    * np.cos(self.agent.direction + i * np.pi / (n_sensors / 2)),
                    self.agent.pos[1]
                    + sensor_data_landmarks[i][0]
                    * np.sin(self.agent.direction + i * np.pi / (n_sensors / 2)),
                ),
                width=2,
            )


"""'
            pygame.draw.line(
                window,
                c,
                self.agent.pos,q
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

\
            if show_text:
                font = pygame.font.Font(None, 24)
                text = font.render(str(int(sensor_data[i][0])), True, "black")
                window.blit(
                    text,
                    (
                        self.agent.pos[0]
                        + sensor_data[i][0]
                        * np.cos(self.agent.direction + i * np.pi / (n_sensors / 2)),
                        self.agent.pos[1]
                        + sensor_data[i][0]
                        * np.sin(self.agent.direction + i * np.pi / (n_sensors / 2)),
                    ),
                )

    def draw_wall_coordinates(self, window):
        for cordinates in self.coordinates:
            pygame.draw.rect(
                window, "red", pygame.Rect(cordinates[0][0], cordinates[0][1], 1, 1)
            )"""
