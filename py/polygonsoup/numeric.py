'''
  _   _   _   _   _   _   _   _   _   _   _
 / \ / \ / \ / \ / \ / \ / \ / \ / \ / \ / \
( P | O | L | Y | G | O | N | S | O | U | P )
 \_/ \_/ \_/ \_/ \_/ \_/ \_/ \_/ \_/ \_/ \_/

Plotter-friendly graphics utilities
© Daniel Berio (@colormotor) 2021 - ...

numeric - numpy-based utilities
'''

import numpy as np

def randspace(a, b, n, minstep=0.1, maxstep=0.6):
    ''' Generate a sequence from a to b with random steps
        minstep and maxstep define the step magnitude
        '''
    v = minstep + np.random.uniform(size=(n-1))*(maxstep-minstep)
    v = np.hstack([[0.0], v])
    v = v / np.sum(v)
    v = np.cumsum(v)
    return a + v*(b-a)

def rank(A, eps=1e-12):
    u, s, vh = np.linalg.svd(A)
    return len([x for x in s if abs(x) > eps])

def weighted_average(v, w):
    v = np.array(v)
    w = np.array(w)
    return np.sum(w*v) / np.sum(w)
