import numpy as np
from math import pi
from scipy.stats import uniform

#Author: Enrico Cavinato 

# agent specifics
X_START, Y_START = 300.0, 300.0
AGENT_SIZE, AGENT_COLOR = 20, "orange"
HEIGHT,WIDTH = 1000,1000
W1,W2,W3 = 0.5, 0.3, 0.2
INSTANTS = 1000

# landmars and walls
LANDMARKS, LANDMARK_COLOR, LANDMARK_SIZE = 2, "black", 2
WALLS_TXT, LANDMARK_TXT = "./data/walls.txt", "./data/landmarks_experiment.txt"


# environment specifics
GAME_RES = WIDTH, HEIGHT = 1300, 700
FPS = 60
GAME_TITLE, SCREEN_COLOR = "SELF NAVIGATION", "white"
