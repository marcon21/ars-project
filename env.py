import pygame
from pygame.locals import *
from typing import List
from actors import Agent, Wall, Landmark
from utils import intersection, distance_from_wall, intersection_line_circle, angle_from_vector
import numpy as np
from copy import deepcopy
import math
from math import cos,sin,degrees

# Author: Daniel Marcon an Aurora Pia Ghiardelli
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
        self.agent.rotate(self.agent.turn_direction / 10)

    def get_sensor_data(self, n_sensors=8, max_distance=200):
        sensor_data = []
        position = self.agent.pos
        x,y = position
        
        #iterate sensors 
        for i in range(n_sensors):
            current_angle = self.agent.direction + i * np.pi / (n_sensors / 2)
            sensor = Wall(x, y, x + max_distance * np.cos(current_angle), y + max_distance * np.sin(current_angle),)

            d = max_distance
            int_point, orientation, signature = None, None, None
            
            #find measurements from walls
            for wall in self.walls:
                intersection_point = intersection(sensor, wall)
                if intersection_point:
                    distance = np.linalg.norm(intersection_point - position)
                    if distance < d:
                        d, int_point, orientation = distance, intersection_point, current_angle
                        
            #find measurements from landmarks            
            for l in self.landmarks:
                intersection_point = intersection_line_circle(sensor, l)

                if intersection_point:
                    for i in intersection_point:
                        # is intersection point on the sensor?
                        if ( np.dot(i-position,sensor.end - position) > 0):
                            distance = np.linalg.norm(i-position)
                            if distance < d:
                                d, int_point, orientation,signature = distance, i, current_angle, l.signature

            sensor_data.append(
                (int_point, (d, orientation, signature), (sensor.start, sensor.end))
            )
        # print(sensor_data)

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
            
    def load_landmarks(self, filename, size, color):
        with open(filename, "r") as f:
            for i, line in enumerate(f, start=1):
                x, y = map(int, line.split())
                landmark = Landmark(x, y, size, i, color)
                self.add_landmark(landmark)
    
    #TODO: legge sensori, forward alla rete, gira le ruote            
    def think(self):
        pass
    
    #TODO: stima la funzione di fintess
    def fitness_score(self)-> float:
        
        #w1* terrain_explored + w2* distanza da muri ( min  sum distances) + w3 * avoidance ( e^-noggetti toccati)
        #tenere traccia
        pass
    
    @property
    def explored_terrain(self)-> float:
        pass

#Authors: we worked toghether on this 
class PygameEnviroment(Enviroment):
    def __init__(self, agent: Agent, walls: List[Wall] = [], color="black"):
        super().__init__(agent, walls=walls)

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
        
        #Draw agent orientation
        pygame.draw.line(window, "orange", (self.agent.pos[0], self.agent.pos[1]),
                 (self.agent.pos[0] + 100 * math.cos(angle_from_vector(self.agent.direction_vector)),
                  self.agent.pos[1] + 100 * math.sin(angle_from_vector(self.agent.direction_vector))), 2)
        
        

        pygame.draw.lines(window, "black", False, self.agent.path, 2)
        self.agent.path = self.agent.path[-1000:]

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
        """
        for i in range(500):
            temp_agent.apply_vector(temp_agent.direction_vector * temp_agent.move_speed)
            temp_agent.rotate(temp_agent.turn_direction / 10)
            next_pos = (temp_agent.pos[0], temp_agent.pos[1])
            if i % 3 == 0:
                pygame.draw.circle(window, "blue", next_pos, 1)
            path.append(next_pos)
        """
        # pygame.draw.lines(window, "blue", False, path, width=2)

        # Draw landmarks
        for landmark in self.landmarks:
            pygame.draw.circle(window, landmark.color, landmark.pos, landmark.size)
            #draw landmark positions
            window.blit(pygame.font.Font(None, 15).render(f"({landmark.pos[0]}, {landmark.pos[1]}), {landmark.signature}", True, "green"), (landmark.pos[0], landmark.pos[1]))


    def draw_sensors(self, window, show_text=False):
        sensor_data = self.get_sensor_data(
            n_sensors=self.agent.n_sensors, max_distance=self.agent.max_distance
        )

        for i in range(self.agent.n_sensors):
            c = "green"
            if sensor_data[i][0] is not None and sensor_data[i][1][2] is not None:
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

            if show_text and sensor_data[i][1][0] < self.agent.max_distance:
                font = pygame.font.Font(None, 24)
                text = font.render("(" + str(int(sensor_data[i][1][0])) + "," + str(int(degrees(sensor_data[i][1][1]))) + ")", True, "black")

                window.blit(
                    text,
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
                )
