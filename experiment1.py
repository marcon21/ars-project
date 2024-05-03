import numpy as np
from math import pi
from scipy.stats import uniform



#agent specifics
X_START,Y_START = 300.,300.
ROTATION_SIZE = pi / 180 * 2
BASE_MOVE_SPEED = 10 
AGENT_SIZE, AGENT_COLOR = 20, "orange"
SENSORS,RANGE = 10, 100
MEAN = np.array([3, 3, 3])
COV_MATRIX = np.diag([1, 1, 0])
R = np.diag([10, 20, 1])
Q = np.diag([20, 10, 1])

#landmars and walls
LANDMARKS, LANDMARK_COLOR, LANDMARK_SIZE  = 2, "black", 20
WALLS_TXT, LANDMARK_TXT = "walls.txt", "landmarks_experiment.txt"


#environment specifics
GAME_RES = WIDTH, HEIGHT = 1000,1000
FPS = 60
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
""".format(ROTATION_SIZE * 180/pi, ROTATION_SIZE * 180/pi)

