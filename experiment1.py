import numpy as np
from math import pi


GAME_RES = WIDTH, HEIGHT = 700,700
FPS = 10
GAME_TITLE, SCREEN_COLOR = "ARS", "white"

ROTATION_SIZE = pi / 180 * 30
BASE_MOVE_SPEED = 0
AGENT_SIZE = 20
SENSORS = 10
RANGE = 100
AGENT_COLOR = "green"
MEAN = np.array([3, 3, 3])
COV_MATRIX = np.diag([10, 100, 1])
R = np.diag([100, 20, 1])
Q = np.diag([10, 100, 100])

LANDMARKS, LANDMARK_COLOR, LANDMARK_SIZE  = 10, "purple", 20

WALLS_TXT, LANDMARK_TXT = "walls.txt", "landmarks_experiment.txt"



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