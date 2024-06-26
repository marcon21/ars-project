import numpy as np
import math
import random as random
import warnings

# Author: Daniel Marcon
warnings.filterwarnings("ignore", category=RuntimeWarning)


def intersection(seg1, seg2):
    def cross_product(p1, p2):
        return p1[0] * p2[1] - p1[1] * p2[0]

    def subtract(p1, p2):
        return (p1[0] - p2[0], p1[1] - p2[1])

    def collinear(p1, p2, p3):
        return cross_product(subtract(p2, p1), subtract(p3, p1)) == 0

    def between(a, b, c):
        if a[0] != b[0]:
            return min(a[0], b[0]) <= c[0] <= max(a[0], b[0])
        else:
            return min(a[1], b[1]) <= c[1] <= max(a[1], b[1])

    p1, p2 = seg1.start, seg1.end
    p3, p4 = seg2.start, seg2.end

    if collinear(p1, p2, p3) and between(p1, p2, p3):
        return p3
    if collinear(p1, p2, p4) and between(p1, p2, p4):
        return p4
    if collinear(p3, p4, p1) and between(p3, p4, p1):
        return p1
    if collinear(p3, p4, p2) and between(p3, p4, p2):
        return p2

    d1 = cross_product(subtract(p3, p4), subtract(p1, p4))
    d2 = cross_product(subtract(p3, p4), subtract(p2, p4))

    if d1 * d2 >= 0:
        return None

    d3 = cross_product(subtract(p1, p2), subtract(p3, p2))
    d4 = cross_product(subtract(p1, p2), subtract(p4, p2))

    if d3 * d4 >= 0:
        return None

    denom = cross_product(subtract(p1, p2), subtract(p3, p4))

    if denom == 0:
        return None

    t = cross_product(subtract(p3, p1), subtract(p3, p4)) / denom

    return (p1[0] - t * (p2[0] - p1[0]), p1[1] - t * (p2[1] - p1[1]))


def distance_from_wall(wall, point, coords=False):
    x1, y1 = wall.start
    x2, y2 = wall.end

    A = point[0] - x1
    B = point[1] - y1
    C = x2 - x1
    D = y2 - y1

    dot = A * C + B * D
    len_sq = C * C + D * D
    param = -1
    if len_sq != 0:
        param = dot / len_sq

    xx = 0
    yy = 0

    if param < 0:
        xx = x1
        yy = y1
    elif param > 1:
        xx = x2
        yy = y2
    else:
        xx = x1 + param * C
        yy = y1 + param * D

    dx = point[0] - xx
    dy = point[1] - yy

    if coords:
        return xx, yy

    return np.sqrt(dx * dx + dy * dy)


def distance(p1, p2):
    return np.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)


def angle_from_vector(vector):
    return math.atan2(vector[1], vector[0])


def intersection_line_circle(wall, circle):
    y = wall.start[1] - circle.pos[1]
    x = wall.start[0] - circle.pos[0]
    dx = wall.end[0] - wall.start[0]
    dy = wall.end[1] - wall.start[1]
    dr = np.sqrt(dx**2 + dy**2)
    D = x * dy - y * dx
    delta = circle.size**2 * dr**2 - D**2

    if delta < 0:
        return None

    sgn = lambda x: 1 if x >= 0 else -1
    sign = sgn(dy)
    if dy == 0:
        sign = sgn(dx)
    x1 = (D * dy + sign * dx * np.sqrt(delta)) / dr**2
    y1 = (-D * dx + abs(dy) * np.sqrt(delta)) / dr**2
    x2 = (D * dy - sign * dx * np.sqrt(delta)) / dr**2
    y2 = (-D * dx - abs(dy) * np.sqrt(delta)) / dr**2

    return (x1 + circle.pos[0], y1 + circle.pos[1]), (
        x2 + circle.pos[0],
        y2 + circle.pos[1],
    )


def create_pairs(pop, fertility_rate=0.5):
    pairs = []
    random.shuffle(pop)

    for i in range(0, int(fertility_rate * len(pop)), 2):  # Create pairs of agents
        pair = (pop[i], pop[i + 1])
        pairs.append(pair)

    return pairs
