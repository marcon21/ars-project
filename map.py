import numpy as np
from math import pi


GAME_RES = WIDTH, HEIGHT = 700,700
FPS = 60
GAME_TITLE = "ARS"

SCREEN_COLOR = "white"
ROTATION_SIZE = pi / 180 * 90
BASE_MOVE_SPEED = 50
AGENT_SIZE = 20
SENSORS = 20
RANGE = 100
AGENT_COLOR = "green"
MEAN = np.array([3, 3, 3])
COV_MATRIX = np.diag([20, 20, 20])
R = np.diag([10, 10, 10])
Q = np.diag([1, 1, 1])
LANDMARKS = 10
LANDMARK_COLOR = "purple"
LANDMARK_SIZE = 20
WALLS_TXT = "walls.txt"
LANDMARK_TXT = "landmarks.txt"


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