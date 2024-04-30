import numpy as np
import sympy
from sympy import Segment, Circle
from sympy import Point2D


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


def circle_line_intersection(circle_center, circle_radius, seg_start, seg_end):
    """
    Trova i punti di intersezione tra un cerchio e un segmento di retta.

    Args:
    circle_center: Tuple, coordinate del centro del cerchio.
    circle_radius: float, raggio del cerchio.
    seg_start: Tuple, coordinate del punto di inizio del segmento.
    seg_end: Tuple, coordinate del punto di fine del segmento.

    Returns:
    List: Lista di tuple contenenti le coordinate dei punti di intersezione.
    """
    cx, cy = circle_center
    r = circle_radius
    x1, y1 = seg_start
    x2, y2 = seg_end

    dx = x2 - x1
    dy = y2 - y1
    fx = x1 - cx
    fy = y1 - cy

    a = dx**2 + dy**2
    b = 2 * (dx * fx + dy * fy)
    c = fx**2 + fy**2 - r**2

    discriminant = b**2 - 4 * a * c

    if discriminant < 0:
        # Nessuna intersezione
        return []
    elif discriminant == 0:
        # Un punto di intersezione
        t = -b / (2 * a)
        x = x1 + t * dx
        y = y1 + t * dy
        return [(x, y)]
    else:
        # Due punti di intersezione
        t1 = (-b + np.sqrt(discriminant)) / (2 * a)
        t2 = (-b - np.sqrt(discriminant)) / (2 * a)
        x1 = x1 + t1 * dx
        y1 = y1 + t1 * dy
        x2 = x1 + t2 * dx
        y2 = y1 + t2 * dy

        return (x1, y1)
