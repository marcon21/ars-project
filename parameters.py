import numpy as np
from math import pi
from scipy.stats import uniform
from torch.nn import functional as F

# Author: Enrico Cavinato

# agent specifics
AGENT_NUMBER = 128 * 2
X_START, Y_START = 500, 500
N_SENSORS, MAX_DISTANCE = 36, 200
AGENT_SIZE, AGENT_COLOR = 8, "orange"
W1, W2, W3 = 0.95, 0.05, 0
INSTANTS = 500
ACTIVATION = F.relu

# landmars and walls
LANDMARKS, LANDMARK_COLOR, LANDMARK_SIZE = 2, "black", 2
WALLS_TXT, LANDMARK_TXT = "./data/walls2.txt", "./data/landmarks_experiment.txt"


# environment specifics
GAME_RES = WIDTH, HEIGHT = 1000, 1000
FPS = 60
GAME_TITLE, SCREEN_COLOR = "SELF NAVIGATION", "white"


# evolution specifics
GENERATIONS = 5
INPUT_SIZE = N_SENSORS
HIDDEN_SIZE = 64
HIDDEN_SIZE2 = 16
OUTPUT_SIZE = 2
GRIDSIZE = 30
