#!/usr/bin/env python3

''' Bezier curve utils '''

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
