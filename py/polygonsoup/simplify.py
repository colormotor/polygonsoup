'''
  _   _   _   _   _   _   _   _   _   _   _
 / \ / \ / \ / \ / \ / \ / \ / \ / \ / \ / \
( P | O | L | Y | G | O | N | S | O | U | P )
 \_/ \_/ \_/ \_/ \_/ \_/ \_/ \_/ \_/ \_/ \_/

Plotter-friendly graphics utilities
© Daniel Berio (@colormotor) 2021 - ...

simplify - Polyline simplification
'''

import numpy as np

from polygonsoup.contrib.dce import dce

# RDP simplification (adapted from https://github.com/fhirschmann/rdp)
def pldist(point, start, end):
    """
    Calculates the distance from ``point`` to the line given
    by the points ``start`` and ``end``.
    :param point: a point
    :type point: numpy array
    :param start: a point of the line
    :type start: numpy array
    :param end: another point of the line
    :type end: numpy array
    """
    if np.all(np.equal(start, end)):
        return np.linalg.norm(point - start)

    return np.divide(
            np.abs(np.linalg.norm(np.cross(end - start, start - point))),
            np.linalg.norm(end - start))

def rdp_rec(M, epsilon, dist=pldist):
    """
    Simplifies a given array of points.
    Recursive version.
    :param M: an array
    :type M: numpy array
    :param epsilon: epsilon in the rdp algorithm
    :type epsilon: float
    :param dist: distance function
    :type dist: function with signature ``f(point, start, end)`` -- see :func:`rdp.pldist`
    """
    dmax = 0.0
    index = -1

    for i in range(1, M.shape[0]):
        d = dist(M[i], M[0], M[-1])

        if d > dmax:
            index = i
            dmax = d

    if dmax > epsilon:
        r1 = rdp_rec(M[:index + 1], epsilon, dist)
        r2 = rdp_rec(M[index:], epsilon, dist)

        return np.vstack((r1[:-1], r2))
    else:
        return np.vstack((M[0], M[-1]))


def _rdp_iter(M, start_index, last_index, epsilon, dist):
    stk = []
    stk.append([start_index, last_index])
    global_start_index = start_index
    indices = np.ones(last_index - start_index + 1, dtype=bool)

    while stk:
        start_index, last_index = stk.pop()

        dmax = 0.0
        index = start_index

        for i in range(index + 1, last_index):
            if indices[i - global_start_index]:
                d = dist(M[i], M[start_index], M[last_index])
                if d > dmax:
                    index = i
                    dmax = d

        if dmax > epsilon:
            stk.append([start_index, index])
            stk.append([index, last_index])
        else:
            for i in range(start_index + 1, last_index):
                indices[i - global_start_index] = False

    return indices

def cleanup_contour(X, eps = 1e-10, closed = False, get_indices=False):
    distfn = lambda p, a, b: max(np.linalg.norm(p - a), np.linalg.norm(p - b))
    return dp_simplify(X, eps, get_indices, closed, distfn);


def dp_simplify(M, eps, get_indices=False, closed=False, dist=pldist):
    ''' Ramer-Douglas-Peucker Simplification adapted from https://github.com/fhirschmann/rdp/'''
    dist=pldist
    if closed:
        M = np.vstack([M, M[0]])

    n = len(M)
    mask = _rdp_iter(M, 0, n - 1, eps, dist)
    I = [i for i in range(n) if mask[i]]

    if closed:
        M = M[:-1]
        I = I[:-1]

    if get_indices:
        return M[I], I
    return M[I]


def vw_simplify(P, tol, closed=False, get_indices=False):
    """ Visvalingam-Whyatt simplification
    from https://github.com/hrishioa/Aviato/blob/bafb2ca8c3d3f11596398e57198a3d62e9a2d39d/app/kartograph/simplify/visvalingam.py
    implementation borrowed from @migurski:
    https://github.com/migurski/Bloch/blob/master/Bloch/__init__.py#L133
    """
    if closed:
        P = np.vstack([P, P[0]])

    if len(P) < 3:
        return P

    min_area = tol ** 2

    I = list(range(len(P)))  # pts stores an index of all non-deleted points

    while len(I) > 4:
        preserved, popped = set(), []
        areas = []

        for i in range(1, len(I) - 1):
            x1, y1 = P[I[i - 1]]
            x2, y2 = P[I[i]]
            x3, y3 = P[I[i + 1]]
            # compute and store triangle area
            areas.append((tri_area(x1, y1, x2, y2, x3, y3), i))

        areas = sorted(areas)

        if not areas or areas[0][0] > min_area:
            # there's nothing to be done
            break

        # Reduce any segments that makes a triangle whose area is below
        # the minimum threshold, starting with the smallest and working up.
        # Mark segments to be preserved until the next iteration.

        for (area, i) in areas:

            if area > min_area:
                # there won't be any more points to remove.
                break

            if i - 1 in preserved or i + 1 in preserved:
                # the current segment is too close to a previously-preserved one.
                #print "-pre", preserved
                continue

            popped.append(i)

            # make sure that the adjacent points
            preserved.add(i - 1)
            preserved.add(i + 1)

        if len(popped) == 0:
            # no points removed, so break out of loop
            break

        popped = sorted(popped, reverse=True)
        for i in popped:
            # remove point from index list
            I = I[:i] + I[i + 1:]

    Q = P[I]
    if closed:
        Q = Q[:-1]
        I = I[:-1]
    if get_indices:
        return Q, I
    return Q

def tri_area(x1, y1, x2, y2, x3, y3):
    """
    computes the area of a triangle given by three points
    implementation taken from:
    http://www.btinternet.com/~se16/hgb/triangle.htm
    """
    return abs((x2*y1-x1*y2)+(x3*y2-x2*y3)+(x1*y3-x3*y1))/2.0
