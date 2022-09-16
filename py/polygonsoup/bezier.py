'''
  _   _   _   _   _   _   _   _   _   _   _
 / \ / \ / \ / \ / \ / \ / \ / \ / \ / \ / \
( P | O | L | Y | G | O | N | S | O | U | P )
 \_/ \_/ \_/ \_/ \_/ \_/ \_/ \_/ \_/ \_/ \_/

Plotter-friendly graphics utilities
© Daniel Berio (@colormotor) 2021 - ...

bezier - Bezier curves
'''

import numpy as np
import matplotlib.pyplot as plt
from scipy.special import binom

def num_bezier(n_ctrl, degree=3):
    return int((n_ctrl - 1) / degree)

def bernstein(n, i):
    bi = binom(n, i)
    return lambda t, bi=bi, n=n, i=i: bi * t**i * (1 - t)**(n - i)

def bezier(P, t, d=0):
    '''Bezier curve of degree len(P)-1. d is the derivative order (0 gives positions)'''
    n = P.shape[0] - 1
    if d > 0:
        return bezier(np.diff(P, axis=0)*n, t, d-1)
    B = np.vstack([bernstein(n, i)(t) for i, p in enumerate(P)])
    return (P.T @ B).T

def bezier_piecewise(Cp, subd=100, degree=3, d=0):
    ''' sample a piecewise Bezier curve given a sequence of control points'''
    num = num_bezier(Cp.shape[0], degree)
    X = []
    for i in range(num):
        P = Cp[i*degree:i*degree+degree+1, :]
        t = np.linspace(0, 1., subd)[:-1]
        Y = bezier(P, t, d)
        X += [Y]
    X = np.vstack(X)
    return X

def compute_beziers(beziers, subd=100, degree=3):
    chain = beziers_to_chain(beziers)
    return bezier_piecewise(chain, subd, degree)

def bezier_at(P, t):
    if len(P)==4:
        return (1.0-t)**3*P[0] + 3*(1.0-t)**2*t*P[1] + 3*(1.0-t)*t**2*P[2] + t**3*P[3]
    else:
        return (1.0-t)**2*P[0] + 2*(1.0-t)*t*P[1] + t**2*P[2]

def plot_control_polygon(Cp, degree=3, lw=0.5, linecolor=np.ones(3)*0.1):
    n_bezier = num_bezier(len(Cp), degree)
    for i in range(n_bezier):
        cp = Cp[i*degree:i*degree+degree+1, :]
        if degree==3:
            plt.plot(cp[0:2,0], cp[0:2, 1], ':', color=linecolor, linewidth=lw)
            plt.plot(cp[2:,0], cp[2:,1], ':', color=linecolor, linewidth=lw)
            plt.plot(cp[:,0], cp[:,1], 'o', color=[0, 0.5, 1.], markersize=4)
        else:
            plt.plot(cp[:,0], cp[:,1], ':', color=linecolor, linewidth=lw)
            plt.plot(cp[:,0], cp[:,1], 'o', color=[0, 0.5, 1.])


def chain_to_beziers(chain, degree=3):
    ''' Convert Bezier chain to list of curve segments (4 control points each)'''
    num = num_bezier(chain.shape[0], degree)
    beziers = []
    for i in range(num):
        beziers.append(chain[i*degree:i*degree+degree+1,:])
    return beziers

def beziers_to_chain(beziers):
    ''' Convert list of Bezier curve segments to a piecewise bezier chain (shares vertices)'''
    n = len(beziers)
    chain = []
    for i in range(n):
        chain.append(list(beziers[i][:-1]))
    chain.append([beziers[-1][-1]])
    return np.array(sum(chain, []))

def spline_to_bezier(tck):
    from scipy.interpolate import insert
    """
    Implementation of Boehm's knot insertion alg, based on https://github.com/zpincus/zplib/blob/930b4b88633c95c7f7761d0183aec882484f00bc/zplib/curve/interpolate.py
    Convert a parametric spline into a sequence of Bezier curves of the same degree.
    Returns a list of Bezier curves of degree k that is equivalent to the input spline.
    Each Bezier curve is an array of shape (k+1,d) where d is the dimension of the
    space; thus the curve includes the starting point, the k-1 internal control
    points, and the endpoint, where each point is of d dimensions."""
    t, c, k = tck
    old_t = np.array(t)
    # the first and last k+1 knots are identical in the non-periodic case, so
    # no need to consider them when increasing the knot multiplicities below
    knots_to_consider = np.unique(t[k+1:-k-1])
    # For each unique knot, bring its multiplicity up to the next multiple of k+1
    # This removes all continuity constraints between each of the original knots,
    # creating a set of independent Bezier curves.
    desired_multiplicity = k+1

    for x in knots_to_consider:
        current_multiplicity = np.sum(old_t == x)
        remainder = current_multiplicity%desired_multiplicity
        if remainder != 0:
            # add enough knots to bring the current multiplicity up to the desired multiplicity
            number_to_insert = desired_multiplicity - remainder
            t, c, k = insert(x, (t, c, k), m=number_to_insert)
            c[0] = c[0][:-k-1] # Need to look into why this is necessary
            c[1] = c[1][:-k-1]

    # group the points into the desired bezier curves
    c = np.array(c).T

    return np.split(c, len(c) // desired_multiplicity)
