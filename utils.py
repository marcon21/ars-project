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
