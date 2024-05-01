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

        for i in range(len(sensor_data)):

            interception = sensor_data[i][0]

            if interception is not None:

                samples = np.random.multivariate_normal(mu, self.Q, 1)[0]

                distance, orientation, signature = sensor_data[i][1]
                sensor_start = sensor_data[i][2][0]
                sensor_end = sensor_data[i][2][1]
                x = interception[0] + (distance) * np.cos(orientation) + samples[0]
                y = interception[1] + (distance) * np.sin(orientation) + samples[1]
                sensor_vector = np.array(sensor_end) - np.array(sensor_start)
                angle = np.arctan2(sensor_vector[1], sensor_vector[0])
                theta = angle + orientation + samples[2]

                return x, y, theta

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
            print("measurments", x, y, theta)
            self.mean = self.mean + np.dot(K, (x, y, theta) - self.mean)
            print("stima", self.mean)
            self.cov_matrix = np.dot(np.eye(3) - np.dot(K, np.eye(3)), self.cov_matrix)

        else:
            self.cov_matrix = np.dot(np.eye(3) - np.dot(K, np.eye(3)), self.cov_matrix)


class PygameKF(Kalman_Filter):
    def __init__(self, env: Enviroment, initial_mean, initial_cov_matrix, R, Q):
        super().__init__(env, initial_mean, initial_cov_matrix, R, Q)

    def show(self, window):
        # print(self.mean[:2])
        pygame.draw.circle(window, "red", self.mean[:2], 10)
