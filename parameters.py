import numpy as np
from math import pi
from scipy.stats import uniform
from torch.nn import functional as F

# Author: Enrico Cavinato

# agent specifics
AGENT_NUMBER = 64
X_START, Y_START = 500, 500
N_SENSORS, MAX_DISTANCE = 12, 200
AGENT_SIZE, AGENT_COLOR = 8, "orange"
W1, W2, W3 = 0.8, 0.2, 0
INSTANTS = 1000
ACTIVATION = F.relu

# landmars and walls
LANDMARKS, LANDMARK_COLOR, LANDMARK_SIZE = 2, "black", 2
WALLS_TXT, LANDMARK_TXT = "./data/walls2.txt", "./data/landmarks_experiment.txt"


# environment specifics
GAME_RES = WIDTH, HEIGHT = 1000, 1000
FPS = 60
GAME_TITLE, SCREEN_COLOR = "SELF NAVIGATION", "white"


# evolution specifics
GENERATIONS = 30
INPUT_SIZE = 12
HIDDEN_SIZE = 32
HIDDEN_SIZE2 = 4
OUTPUT_SIZE = 2
GRIDSIZE = 30
