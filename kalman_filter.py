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
        self.trajectory = [initial_mean[:2]]

    def measurements(self):
        mu = np.array([0, 0, 0])
        sensor_data = self.env.get_sensor_data(
            n_sensors=self.agent.n_sensors, max_distance=self.agent.max_distance)
        l_detections = []
        for perception in sensor_data:

            interception = perception[0]
            signature = perception[1][2]

            if interception is not None and signature is not None:

                # print(sensor_data[i])

                samples = np.random.multivariate_normal(mu, self.Q, 1)[0]
                distance, orientation, signature = perception[1]
                sensor_start = perception[2][0]
                sensor_end = perception[2][1]
                x = self.agent.pos[0] + samples[0]
                y = self.agent.pos[1] + samples[1]
                sensor_vector = np.array(sensor_end) - np.array(sensor_start)
                angle = np.arctan2(sensor_vector[1], sensor_vector[0])
                theta = angle + orientation + samples[2]
                
                l_detections.append((x, y, theta))

        return l_detections
    
    def prediction(self):
        samples = np.random.multivariate_normal(np.zeros(3), self.R, 1)[0]
        movement = self.agent.direction_vector * self.agent.move_speed

        self.mean[0] += movement[0] + samples[0]
        self.mean[1] += movement[1] + samples[1]
        self.mean[2] = np.arctan2(self.agent.direction_vector[1], self.agent.direction_vector[0]) + samples[2]

        self.cov_matrix += self.R

    def correction(self):
        # print("true position", self.agent.pos)
        # print("prediction", self.mean, self.cov_matrix)

        K = np.dot(
            self.cov_matrix,
            np.linalg.inv(self.cov_matrix + self.Q),
        )
        self.prediction()
        meas = self.measurements()

        if meas:

            if len(meas) == 1:
                x = meas[0][0]
                y = meas[0][1]
                theta = meas[0][2]
                # ("measurments", x, y, theta)
                self.mean = self.mean + np.dot(K, (x, y, theta) - self.mean)
                # print("stima", self.mean)
                self.cov_matrix = np.dot(
                    np.eye(3) - np.dot(K, np.eye(3)), self.cov_matrix
                )

            else:
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
                self.mean = self.mean + np.dot(
                    K, (mean_x, mean_y, mean_theta) - self.mean
                )
                self.cov_matrix = np.dot(
                    np.eye(3) - np.dot(K, np.eye(3)), self.cov_matrix
                )

        else:
            self.cov_matrix = np.dot(np.eye(3) - np.dot(K, np.eye(3)), self.cov_matrix)

        self.trajectory.append(self.mean[:2])


class PygameKF(Kalman_Filter):
    def __init__(self, env: Enviroment, initial_mean, initial_cov_matrix, R, Q):
        super().__init__(env, initial_mean, initial_cov_matrix, R, Q)

    def show(self, window):
        # print(self.mean[:2])
        # print(self.trajectory)
        pygame.draw.lines(window, "red", False, self.trajectory, 2)
        """
        trajectory_list = [
            tuple(self.trajectory[i]) for i in range(0, len(self.trajectory), 5)
        ]
        print(trajectory_list)

        if len(self.trajectory) > 6:
            for el in trajectory_list:
                pygame.draw.circle(window, "black", el, 1)

        if len(self.trajectory) >= 200:
            for i in range(0, len(self.trajectory), 200):

                pygame.draw.ellipse(
                    window,
                    "red",
                    (
                        self.trajectory[i][0],
                        self.trajectory[i][1],
                        self.cov_matrix[0][0],
                        self.cov_matrix[1][1],
                    ),
                )
        """
