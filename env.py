import pygame
from pygame.locals import *
from typing import List
from actors import Agent, Wall, Landmark
from utils import intersection, distance_from_wall, intersection_line_circle
import numpy as np
from copy import deepcopy
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

    def move_agent(self, dt=1 / 60):
        self.move_vector = self.agent.direction_vector * self.agent.move_speed
        for wall in self.walls:
            current_d = distance_from_wall(wall, self.agent.pos)

            if current_d <= self.agent.size:
                # Vector of the wall direction
                wall_vector = np.array(
                    [wall.end[0] - wall.start[0], wall.end[1] - wall.start[1]]
                )
                wall_vector = wall_vector / np.linalg.norm(wall_vector)

                # Vector of the agent parallel to the wall
                parallel_component = np.dot(wall_vector, self.move_vector) * wall_vector

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
                    self.move_vector = parallel_component

        # Check if the agent is making an illegal move
        for wall in self.walls:
            intersection_point = intersection(
                Wall(
                    self.agent.pos[0],
                    self.agent.pos[1],
                    self.agent.pos[0] + self.move_vector[0],
                    self.agent.pos[1] + self.move_vector[1],
                ),
                wall,
            )
            if intersection_point:
                print("ILLEGAL MOVE")
                return

        self.agent.apply_vector(self.move_vector)
        self.agent.rotate(self.agent.turn_direction / 10)

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
            orientation = None
            signature = None
            for wall in self.walls:
                intersection_point = intersection(sensor, wall)
                if intersection_point:
                    distance = np.linalg.norm(
                        np.array(intersection_point) - np.array(self.agent.pos)
                    )
                    if distance < d:
                        d = distance
                        int_point = intersection_point
                        orientation = current_angle
            for l in self.landmarks:
                intersection_point = intersection_line_circle(sensor, l)

                if intersection_point:
                    for i in intersection_point:
                        # is intersection point on the sensor?
                        if (
                            np.dot(
                                np.array(i) - np.array(self.agent.pos),
                                np.array(sensor.end) - np.array(self.agent.pos),
                            )
                            > 0
                        ):
                            distance = np.linalg.norm(
                                np.array(i) - np.array(self.agent.pos)
                            )
                            if distance < d:
                                d = distance
                                int_point = i
                                orientation = current_angle
                                signature = l.signature

            sensor_data.append((int_point, (d, orientation, signature)))
        return sensor_data

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


class Kalman_Filter:

    def __init__(self, env: Enviroment, initial_mean, initial_cov_matrix, R, Q):
        self.env = env
        self.agent = env.agent
        self.mean = initial_mean
        self.cov_matrix = initial_cov_matrix
        self.R = R
        self.Q = Q

    def measurements(self):
        mu = np.array([0, 0, 0])
        sensor_data = self.env.get_sensor_data(self.agent.n_sensors)
        for el in sensor_data:

            if el[0] != None:
                samples = np.random.multivariate_normal(mu, self.Q, 1)[0]

                distance, orientation, signature = el[1]
                x = el[0][0] + (el[1][0]) * np.cos(orientation) + samples[0]
                y = el[0][1] + (el[1][0]) * np.sin(orientation) + samples[1]
                sensor_vector = el[2][1] - el[2][0]
                angle = np.arctan2(sensor_vector[1], sensor_vector[0])
                theta = angle + orientation + samples[2]

                return x, y, theta
            else:
                return None

    def prediction(self):
        mu = np.array([0, 0, 0])
        samples = np.random.multivariate_normal(mu, self.R, 1)[0]
        self.mean[0] = (
            self.mean[0]
            + self.agent.direction_vector * self.agent.move_speed[0]
            + samples[0]
        )
        self.mean[1] = (
            self.mean[1]
            + self.agent.direction_vector * self.agent.move_speed[1]
            + samples[1]
        )
        self.mean[2] = (
            np.arctan2(self.agent.direction_vector[1], self.agent.direction_vector[0])
            + samples[2]
        )
        self.cov_matrix = self.cov_matrix + self.R

    def correction(self):
        print("true position", self.agent.pos)
        print("prediction", self.mean, self.cov_matrix)
        print("measurments", x, y, theta)

        K = np.dot(
            self.cov_matrix,
            np.dot(
                np.eye(3).T,
                np.linalg.inv(
                    np.dot(np.eye(3), np.dot(self.cov_matrix, np.eye(3).T) + self.Q)
                ),
            ),
        )
        self.prediction()
        meas = self.measurements()
        if meas:
            x, y, theta = meas
            self.mean = self.mean + np.dot(K, (x, y, theta) - self.mean)
            self.cov_matrix = np.dot(np.eye(3) - np.dot(K, np.eye(3)), self.cov_matrix)
            print("correction", self.measurements)
        else:
            self.cov_matrix = np.dot(np.eye(3) - np.dot(K, np.eye(3)), self.cov_matrix)


class PygameEnviroment(Enviroment):
    def __init__(
        self, agent: Agent, kf: Kalman_Filter, walls: List[Wall] = [], color="black"
    ):
        super().__init__(agent, walls=walls)
        self.kf = kf

        pass

    def show(self, window):

        for wall in self.walls:

            pygame.draw.line(window, "black", wall.start, wall.end, width=5)

        agent_color = self.agent.color
        # for wall in self.walls:
        #     dist = distance_from_wall(wall, self.agent.pos)
        #     if dist == self.agent.size:
        #         agent_color = "blue"
        #     if dist < self.agent.size:
        #         agent_color = "red"

        # Draw agent
        pygame.draw.circle(window, agent_color, self.agent.pos, self.agent.size)
        for point in self.agent.path:
            index = self.agent.path.index(point)
            final_index = len(self.agent.path) - 1
            if index > 0 and index < final_index:
                pygame.draw.lines(
                    window, "black", False, [point, self.agent.path[index + 1]], 2
                )

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
        temp_agent = deepcopy(self.agent)
        for i in range(500):
            temp_agent.apply_vector(temp_agent.direction_vector * temp_agent.move_speed)
            temp_agent.rotate(temp_agent.turn_direction / 10)
            next_pos = (temp_agent.pos[0], temp_agent.pos[1])
            if i % 3 == 0:
                pygame.draw.circle(window, "blue", next_pos, 1)
            path.append(next_pos)
        # pygame.draw.lines(window, "blue", False, path, width=2)

        # Draw landmarks
        for landmark in self.landmarks:
            pygame.draw.circle(window, landmark.color, landmark.pos, landmark.size)

    def draw_sensors(self, window, n_sensors=10, max_distance=200, show_text=False):
        sensor_data = self.get_sensor_data(
            n_sensors=n_sensors, max_distance=max_distance
        )
        for i in range(n_sensors):
            c = "green"
            if sensor_data[i][0] is not None:
                c = "red"
                pygame.draw.circle(window, "red", sensor_data[i][0], 5)

            pygame.draw.line(
                window,
                c,
                self.agent.pos,
                (
                    self.agent.pos[0]
                    + sensor_data[i][1][0]
                    * np.cos(self.agent.direction + i * np.pi / (n_sensors / 2)),
                    self.agent.pos[1]
                    + sensor_data[i][1][0]
                    * np.sin(self.agent.direction + i * np.pi / (n_sensors / 2)),
                ),
                width=2,
            )

            if show_text:
                font = pygame.font.Font(None, 24)
                text = font.render(str(int(sensor_data[i][0][0])), True, "black")
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
