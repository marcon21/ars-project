import numpy as np
from math import pi
from scipy.stats import uniform
from torch.nn import functional as F

# Author: Enrico Cavinato

# agent specifics
AGENT_NUMBER = 20
X_START, Y_START = 350.0, 300.0
N_SENSORS, MAX_DISTANCE = 12, 200
AGENT_SIZE, AGENT_COLOR = 10, "orange"
W1, W2, W3 = 0.9, 0.05, 0.05
INSTANTS = 1000
ACTIVATION = F.relu

# landmars and walls
LANDMARKS, LANDMARK_COLOR, LANDMARK_SIZE = 2, "black", 2
WALLS_TXT, LANDMARK_TXT = "./data/walls.txt", "./data/landmarks_experiment.txt"


# environment specifics
GAME_RES = WIDTH, HEIGHT = 1000, 900
FPS = 200
GAME_TITLE, SCREEN_COLOR = "SELF NAVIGATION", "white"


# evolution specifics
GENERATIONS = 20
INPUT_SIZE = 12
HIDDEN_SIZE = 32
HIDDEN_SIZE2 = 4
OUTPUT_SIZE = 2
