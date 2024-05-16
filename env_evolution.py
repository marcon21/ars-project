from env import Enviroment
from evolved_agent import EvolvedAgent
from actors import Agent, Wall, Landmark
from typing import List
import numpy as np  
import pygame
from copy import deepcopy
from math import degrees, atan2, pi
from utils import distance_from_wall, intersection, angle_from_vector
import math
import torch

class EnvEvolution(Enviroment):    
    
    '''
    Class that represents the enviroment for the evolution of the agent
    
    Attributes: 
        agent (EvolvedAgent): the agent that will evolve
        walls (List[Wall]): the walls of the enviroment
        landmarks (List[Landmark]): the landmarks of the enviroment
        height (int): the height of the enviroment
        width (int): the width of the enviroment
        map (np.array): the map of the enviroment
        collisions (int): the number of collisions of the agent
        movements (int): the number of movements of the agent
        instants (int): the number of instants of the enviroment
        W1 (float): the weight of the explored terrain in the fitness score
        W2 (float): the weight of the distance in the fitness score
        W3 (float): the weight of the collisions in the fitness score
        distance (np.array): the distance of the agent from the walls in the instants
    '''
    def __init__(self, agent: EvolvedAgent, walls: List[Wall] = [], landmarks: List[Landmark] = [], height=800, width=800,instants=1000, W1=0.5, W2=0.3, W3=0.2):
        super().__init__(agent, walls, landmarks)
        self.height = height
        self.width = width
        self.map = np.zeros((self.height//10, self.width//10))
        self.collisions = 0
        self.movements = 0
        self.instants = instants
        self.W1 = W1        
        self.W2 = W2
        self.W3 = W3
        self.distance = self.agent.max_distance * np.ones(self.instants)
            
    #TODO: legge sensori, forward alla rete, gira le ruote   
    # (int_point, (d, orientation, signature), (sensor.start, sensor.end))         
    def think(self):
        '''
        Think function of the agent
        
        Returns:
            vl (float): the left velocity of the agent
            vr (float): the right velocity of the agent
        '''
        self.movements += 1
        self.map[self.agent.pos[0]//10, self.agent.pos[1]//10] = 1 
        sensor_data = self.get_sensor_data(self.agent.n_sensors,self.agent.max_distance)
        for data in sensor_data:
            if data[1][0] < self.agent.size:
                self.collisions += 1
        self.distance[self.movements] = np.min([data[1][0] for data in sensor_data])  #REVIEW 
        distances = [float(data[1][0]) for data in sensor_data]
        print(distances)
        vl,vr = self.agent.controller.forward(torch.tensor(distances))
        return vl,vr
        
    def move_agent(self, dt=1 / 60):
        vl,vr = self.think()
        ds = dt * (vl + vr) / 2
        dx,dy = np.cos(ds), np.sin(ds)  
        dtheta = dt * (vr - vl) / self.agent.size  
        move_vector = self.agent.direction_vector + np.array([round(dx),round(dy)])

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
        
    def fitness_score(self)-> float:
        return self.W1 * self.explored_terrain + self.W2 * np.mean(self.distance) + self.W3 * np.exp(-self.collisions)
    
    @property
    def explored_terrain(self)-> float:
        return np.sum(self.map) / (self.height//10 * self.width//10)
    
    
class PygameEvolvedEnviroment(EnvEvolution):

    def __init__(self, agent: Agent, walls: List[Wall] = [], color="black", landmarks: List[Landmark] = [], height=800, width=800, instants=1000, w1=0.5, w2=0.3, w3=0.2):
        super().__init__(agent, walls=walls, landmarks=[], height=height, width=width, instants=instants, W1=w1, W2=w2, W3=w3)

    def show(self, window):
        for wall in self.walls:
            pygame.draw.line(window, "black", wall.start, wall.end, width=5)

        agent_color = self.agent.color
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