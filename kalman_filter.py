from env import Enviroment, PygameEnviroment
from actors import Agent
import numpy as np
import pygame
from math import sin, cos, radians, degrees, sqrt
from pygame.locals import *
from utils import angle_from_vector
from numpy.random import multivariate_normal
from archive.parameters import TRAJ

#Authors: Aurora Pia Ghiardelli, Daniel Marcon, Enrico Cavinato
class Kalman_Filter:
    def __init__(self, env: Enviroment, initial_mean, initial_cov_matrix, R, Q):
        self.env = env
        self.agent = env.agent
        self.mean = initial_mean
        self.cov_matrix = initial_cov_matrix
        self.R = R
        self.Q = Q
        self.trajectory = [initial_mean[:2]]
        self.mean_prediction = initial_mean
        self.cov_prediction = initial_cov_matrix
        self.traj = TRAJ

    # sensor data: [(int_point, (d, orientation, signature), (sensor.start, sensor.end))]
    def measurements(self):
        mu = np.array([0, 0, 0])
        sensor_data = self.env.get_sensor_data(
            n_sensors=self.agent.n_sensors, max_distance=self.agent.max_distance
        )
        l_detections = []
        for perception in sensor_data:

            interception = perception[0]
            signature = perception[1][2]

            if interception is not None and signature is not None:

                # print(sensor_data[i])

                samples = multivariate_normal(mu, self.Q, 1)[0]
                distance, orientation, signature = perception[1]
                sensor_start, sensor_end = perception[2]

                x = self.agent.pos[0] + samples[0]
                y = self.agent.pos[1] + samples[1]

                sensor_vector = np.array(sensor_end) - np.array(sensor_start)
                theta = angle_from_vector(sensor_vector) + orientation + samples[2]

                # if self.traj:
                #     self.traj -= 1
                # else:
                l_detections.append((x, y, theta))
                # self.traj += TRAJ

        return l_detections

    def prediction(self):
        mu = np.array([0, 0, 0])
        samples = multivariate_normal(mu, self.R, 1)[0]
        self.mean[0] = (
            self.mean[0]
            + (self.agent.direction_vector * self.agent.move_speed)[0]
            + samples[0]
        )
        self.mean[1] = (
            self.mean[1]
            + (self.agent.direction_vector * self.agent.move_speed)[1]
            + samples[1]
        )
        self.mean[2] = angle_from_vector(self.agent.direction_vector) + samples[2]
        self.cov_matrix = self.cov_matrix + self.R
        self.mean_prediction, self.cov_prediction = self.mean, self.cov_matrix

    def correction(self):
        # print("true position", self.agent.pos)
        # print("prediction", self.mean, self.cov_matrix)

        self.prediction()
        meas = self.measurements()

        if meas:
            if len(meas) >= 2:
                K = np.dot(
                    self.cov_matrix,
                    np.linalg.inv(self.cov_matrix + self.Q),
                )
                sum_x = 0
                sum_y = 0
                sum_theta = 0
                for tupla in meas:
                    sum_x += tupla[0]
                    sum_y += tupla[1]
                    sum_theta += tupla[2]
                mean_x = sum_x / len(meas)
                mean_y = sum_y / len(meas)
                mean_theta = sum_theta / len(meas)
                # ("measurments", x, y, theta)
                self.mean = self.mean + np.dot(
                    K, (mean_x, mean_y, mean_theta) - self.mean
                )
                # print("stima", self.mean)
                self.cov_matrix = np.dot(
                    np.eye(3) - np.dot(K, np.eye(3)), self.cov_matrix
                )
            else:

                mean = np.array([self.mean[0], self.mean[1]])
                K = np.dot(
                    self.cov_matrix,
                    np.linalg.inv(self.cov_matrix + self.Q),
                )
                x = meas[0][0]
                y = meas[0][1]

                new_mean = mean + np.dot(K[:2, :2], np.array([x, y]) - mean)
                self.mean[0] = new_mean[0]
                self.mean[1] = new_mean[1]
                self.cov_matrix = np.dot(
                    np.eye(3) - np.dot(K, np.eye(3)), self.cov_matrix
                )

        else:
            K = np.dot(
                self.cov_matrix,
                np.linalg.inv(self.cov_matrix + self.Q),
            )
            self.cov_matrix = np.dot(np.eye(3) - np.dot(K, np.eye(3)), self.cov_matrix)

        self.trajectory.append([self.mean[0], self.mean[1]])

#Auythors: ENrico Cavinato, Daniel Marcon
class PygameKF(Kalman_Filter):
    def __init__(self, env: Enviroment, initial_mean, initial_cov_matrix, R, Q):
        super().__init__(env, initial_mean, initial_cov_matrix, R, Q)

    def show(self, window):
        # print(self.mean[:2])
        # print(self.trajectory)

        def draw_semi_transparent_ellipse(
            screen, alpha, pose, horizontal_radius, vertical_radius
        ):
            surface = pygame.Surface(
                (horizontal_radius * 2, vertical_radius * 2), pygame.SRCALPHA
            )
            pygame.draw.ellipse(
                surface,
                (100, 100, 100, alpha),
                (0, 0, horizontal_radius * 2, vertical_radius * 2),
            )
            rotated_surface = pygame.transform.rotate(surface, degrees(pose[2]))
            screen.blit(
                rotated_surface,
                (pose[0] - horizontal_radius, pose[1] - vertical_radius),
            )

        # after prediction
        draw_semi_transparent_ellipse(
            window, 200, self.mean, sqrt(self.cov_matrix[0][0]), sqrt(self.cov_matrix[1][1])
        )
        pygame.draw.line(
            window,
            "blue",
            (self.env.agent.pos[0], self.env.agent.pos[1]),
            (
                self.env.agent.pos[0] + 100 * cos(self.mean[2]),
                self.env.agent.pos[1] + 100 * sin(self.mean[2]),
            ),
            2,
        )

        # before prediction
        draw_semi_transparent_ellipse(
            window,
            50,
            self.mean_prediction,
            sqrt(self.cov_prediction[0][0]),
            sqrt(self.cov_prediction[1][1]),
        )
        pygame.draw.line(
            window,
            "grey",
            (self.env.agent.pos[0], self.agent.pos[1]),
            (
                self.env.agent.pos[0] + 100 * cos(self.mean_prediction[2]),
                self.env.agent.pos[1] + 100 * sin(self.mean_prediction[2]),
            ),
            2,
        )

        # print(self.trajectory[-3:])
        for point in self.trajectory[-300:]:
            pygame.draw.circle(window, "purple", point, 1)

        # if len(self.trajectory) >= 5:
        # pygame.draw.lines(window, "black", False, self.trajectory[-500:], 1)
