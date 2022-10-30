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
from scipy.ndimage.filters import gaussian_filter1d

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

def dtw(x, y, w = np.inf, get_dist = False, distfn = lambda a, b: np.dot(b-a, b-a)):
    nx = len(x)
    ny = len(y)
    w = max(w, abs(nx-ny))

    D = np.ones((nx, ny))*np.inf
    D[0,0] = 0

    for i in range(nx):
        for j in range(max(1, i-w), min(ny, i+w)):
            D[i][j] = 0

    for i in range(nx):
        for j in range(max(1, i-w), min(ny, i+w)):
            D[i,j] = distfn(x[i], y[j]) + np.min([D[i - 1][j], D[i][j - 1], D[i - 1][j - 1]])

    if get_dist:
        return D[nx - 1, ny - 1]

    i = nx - 1
    j = ny - 1
    # recover path
    p = [[i,j]]

    while i > 0 and j > 0:
        id = np.argmin([D[i][j - 1], D[i - 1][j], D[i - 1][j - 1]])
        if id == 0:
            j = j - 1
        elif id == 1:
            i = i - 1
        else:
            i = i - 1
            j = j - 1

        p.append([i, j])

    # if max(nx, ny) > len(p):
    #     raise ValueError
    return p[::-1]


def dtw_path(x, y, w = np.inf):
  return dtw(x, y, w, False)

def dtw_dist(x, y, w = np.inf):
  return mth.dtw(x, y, w, True)

def gaussian_smooth(X, sigma, mode='reflect'):
    #mode = 'reflect'
    return gaussian_filter1d(X, sigma=sigma, mode=mode)

def smooth_diff(X, sigma, order=1):
    return gaussian_filter1d(X, sigma, order=order)

# def maxima(X, diffSigma, thresh, eps, offsetThresh):
#     if diffSigma > 0:
#         D = smooth_diff(X, diffSigma);
#     else:
#         D = np.diff(X);
#     MM = []
#     ind = 0

#     THR = lambda a, b: (np.abs(a) + np.abs(b)) > eps
#     #define THR(a, b) ((fabs(a) + fabs(b)) > eps)

#     c = 0
#     for i in range(len(D) - 1):
#         ind += 1
#         d1  = D[i]
#         d2 = D[i + 1]
#         thr = (np.abs(d1) + np.abs(d2))
#         xx = X[i]

#         if (d1 > 0.0 and d2 < 0.0):
#             if (THR(d1, d2) and X[i] > thresh):

#                 c = 0
#                 MM.append(ind)
#       }
#     }
#     c++;
#   }

#   return MM;
# }
