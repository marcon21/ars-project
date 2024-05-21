from actors import Agent
from nn import NN
import math


class EvolvedAgent(Agent):
    def __init__(
        self,
        x=0,
        y=0,
        move_speed=5,
        size=40,
        n_sensors=10,
        max_distance=200,
        color="red",
        controller: NN = None,
        
    ) -> None:
        super().__init__(x, y, move_speed, size, n_sensors, max_distance, color)
        self.controller = controller
        self.genome = self.controller.get_weights()
        # self.terrain_explored = None
        self.min_distance = math.inf
        self.number_obstacles = 0

        # self.fitness_score = -1
