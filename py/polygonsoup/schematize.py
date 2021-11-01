'''
  _   _   _   _   _   _   _   _   _   _   _
 / \ / \ / \ / \ / \ / \ / \ / \ / \ / \ / \
( P | O | L | Y | G | O | N | S | O | U | P )
 \_/ \_/ \_/ \_/ \_/ \_/ \_/ \_/ \_/ \_/ \_/

Plotter-friendly graphics utilities
© Daniel Berio (@colormotor) 2021 - ...

schematize - C-oriented polyline schematization.
Quick and dirty implementation of angular schematization as in:
Dwyer et al. (2008) A fast and simple heuristic for metro map path simplification
https://pdfs.semanticscholar.org/c791/260d13dff6a7fad539ae182e9791c909aa17.pdf
still can produce some very short segments, must fix.
'''

import numpy as np
from polygonsoup.algorithms import DoublyLinkedList
import polygonsoup.geom as geom
import networkx as nx

def vec(*args):
    return np.array(args)

def radians( x ):
    return np.pi/180*x

def rot_2d( theta ):
    m = np.eye(2)

    ct = np.cos(theta)
    st = np.sin(theta)

    m[0,0] = ct
    m[0,1] = -st
    m[1,0] = st
    m[1,1] = ct

    return m

def perp(x):
    return np.dot([[0,-1],[1, 0]], x)

def line_intersection(s0, s1, eps=1e-10):
    sp0 = np.cross(*[np.concatenate([s, [1]]) for s in s0])
    sp1 = np.cross(*[np.concatenate([s, [1]]) for s in s1])
    ins = np.cross(sp0, sp1)
    if abs(ins[2]) < eps:
        return False, np.zeros(2)
    return True, vec(ins[0]/ins[-1], ins[1]/ins[-1])

def project_on_line(p, a, b):
    d = b - a
    t = np.dot(p - a, d) / np.dot(d, d)
    return a + (b - a)*t

def perp(v):
    return np.array([-v[1], v[0]])

def cost(V, n):
    b  = np.mean(V, axis=0)
    bn = np.dot(b, n)
    c  = 0.0
    for v in V:
        ct = np.dot(v, n) - bn
        c += ct * ct
    return c

def best_cost(V, N):
    costs = [cost(V, n) for n in N]
    return np.argmin(costs)

def merge(blocks, b0, b1, N):
    if b0.best != b1.best:
        return b1
    b0.V += b1.V[1:]
    b0.best = best_cost(b0.V, N)
    blocks.remove(b1)
    return b0

def schematize(P, C, angle_offset, closed=False, get_edge_inds=False, maxiter=1000):
    P = list(P)
    G = nx.Graph()
    n = len(P)

    if closed:
        P = P + [P[0]] #, P[1]]

    edges = [(i, i+1) for i in range(n-1)]
    for a, b in edges:
        G.add_edge(a, b)

    # Orthognonal directions
    dtheta = radians(angle_offset)
    N = []
    if type(C) in [list, np.ndarray]:
        for ang in C:
            th = geom.radians(ang) + dtheta
            N.append(vec(np.cos(th), np.sin(th)))
    else:
        for i in range(C):
            th = (np.pi / C)*i + dtheta
            #print(geom.degrees(th))
            N.append(vec(np.cos(th), np.sin(th)))

    V = {}
    best = {}
    blocks = DoublyLinkedList()

    for i, e in enumerate(edges):
        b = DoublyLinkedList.Node()
        b.V = []
        for j in range(2):
            b.V.append(P[e[j]])
        b.best = best_cost(b.V, N)
        b.edge_index = i
        blocks.add(b)

    for iter in range(maxiter):
        b0 = blocks.front
        b1 = b0.next
        num_merges = 0
        while b1:
            if merge(blocks, b0, b1, N) == b0:
                num_merges += 1
                # backtrack
                b1 = b0
                if b1.prev is not None:
                    if merge(blocks, b1.prev, b1, N) == b1.prev:
                        b1 = b1.prev
                        num_merges += 1
            b0 = b1
            b1 = b1.next

        if not num_merges:
            break

    P = []
    edge_inds = []
    block = blocks.front
    prev  = None

    while block is not None:
        b = np.mean(block.V, axis=0)
        n = N[block.best]
        d = perp(n)

        if prev is not None:
            res, ins = line_intersection((b_prev, b_prev + d_prev), (b, b + d))
            if res:
                P.append(ins)
                edge_inds.append(block.edge_index)
        else:
            P.append(project_on_line(block.V[0], b, b + d))
            edge_inds.append(block.edge_index)

        prev   = block;
        b_prev = b
        d_prev = d
        block  = block.next

    P.append(project_on_line(blocks.back.V[-1], b, b + d))
    edge_inds.append(blocks.back.edge_index)
    if closed and len(P) > 2:
        # pass #P.pop()
        res, ins = line_intersection((P[0], P[1]), (P[-1], P[-2]))
        if res:
            P[0] = ins
            P[-1] = ins
        elif len(P) > 3:
            P.pop()
            res, ins = line_intersection((P[0], P[1]), (P[-1], P[-2]))
            if res:
                P[0] = ins
                P[-1] = ins

        #if len(P) > 3:
        #    P.pop()

    P = np.array(P)
    if get_edge_inds:
        return P, edge_inds
    return P


def quantize(v, step ):
    return np.round(v/step)*step

def quantize_rotation_matrix(R, C, dtheta=0):
    if C == 0:
        return R
    x = R[:2, 0]
    th = np.arctan2(x[1], x[0])
    th = quantize(th-dtheta, np.pi/(2*C)) + dtheta
    R = np.array(R)
    R[:2, :2] = rot_2d(th)
    return R
