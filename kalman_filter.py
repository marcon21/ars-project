from env import Enviroment
from actors import Agent
import numpy as np
import pygame
from pygame.locals import *


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
        sensor_data = self.env.get_sensor_data(
            n_sensors=self.agent.n_sensors, max_distance=self.agent.max_distance
        )

        for el in sensor_data:
            if el[0] is not None:
                print(el)

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
            + (self.agent.direction_vector * self.agent.move_speed)[0]
            + samples[0]
        )
        self.mean[1] = (
            self.mean[1]
            + (self.agent.direction_vector * self.agent.move_speed)[1]
            + samples[1]
        )
        self.mean[2] = (
            np.arctan2(self.agent.direction_vector[1], self.agent.direction_vector[0])
            + samples[2]
        )
        self.cov_matrix = self.cov_matrix + self.R

    def correction(self):
        # print("true position", self.agent.pos)
        # print("prediction", self.mean, self.cov_matrix)

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
            # print("measurments", x, y, theta)
            self.mean = self.mean + np.dot(K, (x, y, theta) - self.mean)
            self.cov_matrix = np.dot(np.eye(3) - np.dot(K, np.eye(3)), self.cov_matrix)

        else:
            self.cov_matrix = np.dot(np.eye(3) - np.dot(K, np.eye(3)), self.cov_matrix)


class PygameKF(Kalman_Filter):
    def __init__(self, env: Enviroment, initial_mean, initial_cov_matrix, R, Q):
        super().__init__(env, initial_mean, initial_cov_matrix, R, Q)

    def show(self, window):
        # print(self.mean[:2])
        pygame.draw.circle(window, "red", self.mean[:2], 10)
