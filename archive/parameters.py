import numpy as np
from math import pi
from scipy.stats import uniform

#Author: Enrico Cavinato 

# agent specifics
X_START, Y_START = 300.0, 300.0
ROTATION_SIZE = pi / 180 * 2
BASE_MOVE_SPEED = 20
AGENT_SIZE, AGENT_COLOR = 20, "orange"
SENSORS, RANGE = 100, 10
MEAN = np.array([X_START + 200 // 2, Y_START + 300, 0])
COV_MATRIX = np.diag([1, 1, 1])
R = np.diag([30, 120, 1])
Q = np.diag([1, 1, 0.05])
TRAJ = 1  # each traj point a point is added to trajectory

# landmars and walls
LANDMARKS, LANDMARK_COLOR, LANDMARK_SIZE = 2, "black", 2
WALLS_TXT, LANDMARK_TXT = "./data/walls.txt", "./data/landmarks_experiment.txt"


# environment specifics
GAME_RES = WIDTH, HEIGHT = 1300, 700
FPS = 60
SLIDER_HEIGHT = 5
SLIDER_WIDTH = 100
GAME_TITLE, SCREEN_COLOR = "EXPERIMENT 1: ROBOT CAN SEE 3 LANDMARKS", "white"
INSTRUCTIONS = """
--------------------
Press 'q' or 'ARROW L' to rotate left by {} degrees \n
Press 'e' or 'ARROW R' to rotate right by {} degrees \n
Press 'n' to add a wall \n
Press 's' to save walls to file \n
Press 'l' to load walls from file \n
Press 'BACKSPACE' to remove all walls \n
Press 'r' to reset agent \n
Press 't' to toggle text visibility \n
Press 'SPACE' to toggle movement \n
""".format(
    ROTATION_SIZE * 180 / pi, ROTATION_SIZE * 180 / pi
)
