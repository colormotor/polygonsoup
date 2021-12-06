'''
  _   _   _   _   _   _   _   _   _   _   _
 / \ / \ / \ / \ / \ / \ / \ / \ / \ / \ / \
( P | O | L | Y | G | O | N | S | O | U | P )
 \_/ \_/ \_/ \_/ \_/ \_/ \_/ \_/ \_/ \_/ \_/

Plotter-friendly graphics utilities
© Daniel Berio (@colormotor) 2021 - ...

geom - Geometry utilities
'''

import copy
import numpy as np
from numpy import (sin, cos, tan)
from numpy.linalg import (norm, det, inv)
from scipy.interpolate import interp1d, splprep, splev
import numbers
import pdb

def is_number(x):
    return isinstance(x, numbers.Number)

def is_compound(S):
    '''Returns True if S is a compound polyline,
    a polyline is represented as a list of points, or a numpy array with as many rows as points'''
    if type(S[0])==list:
        return True
    if type(S[0])==np.ndarray and len(S[0].shape) > 1:
        return True
    return False

def is_polyline(P):
    '''A polyline can be represented as either a list of points or a NxDim array'''
    if (type(P[0]) == np.ndarray and
        len(P[0].shape) < 2):
        return True
    return False

def close_path(P):
    if type(P) == list:
        return P + [P[0]]
    return np.vstack([P, P[0]])

def close(S):
    if is_compound(S):
        return [close_path(P) for P in S]
    return close_path(S)

def is_empty(S):
    if type(S)==list and not S:
        return True
    return False

def vec(*args):
    return np.array(args)

def radians( x ):
    return np.pi/180*x

def degrees( x ):
    return x * (180.0/np.pi)

def normalize(v):
    return v / np.linalg.norm(v)

def angle_between(a, b):
    ''' Angle between two vectors (2d)'''
    return np.arctan2( a[0]*b[1] - a[1]*b[0], a[0]*b[0] + a[1]*b[1] )

def distance(a, b):
    return norm(b-a)

def distance_sq(a, b):
    return np.dot(b-a, b-a)

def normc(X):
    ''' Normalizes columns of X'''
    l = np.sqrt( X[:,0]**2 + X[:,1]**2 )
    Y = np.array(X)
    return Y / np.maximum(l, 1e-9).reshape(-1,1)


def normals_2d(P, closed=0, vertex=False):
    ''' 2d normals (fixme)'''
    if closed:
        P = np.vstack([P[-1], P, P[0]])

    D = np.diff(P, axis=0)
    if vertex and D.shape[0] > 1:
        T = D[:-1] + D[1:]
        if not closed:
            T = np.vstack([T[0], T, T[-1]])
        N = np.dot([[0,1],[-1, 0]], T.T).T
    else:
        T = D
        N = np.dot([[0,1],[-1, 0]], T.T).T
        if not closed:
            N = np.vstack([N, N[-1]])
        else:
            N = N[:-1]
    return normc(N)

def point_line_distance(p, a, b):
    if np.array_equal(a,b):
        return distance(p, a)
    else:
        return abs(det(np.array([b-a, a-p]))) / norm(b-a)

def signed_point_line_distance(p, a, b):
    if np.array_equal(a,b):
        return distance(p, a)
    else:
        return (det(np.array([b-a, a-p]))) / norm(b-a)

def point_segment_distance(p, a, b):
    a, b = np.array(a), np.array(b)
    d = b - a
    # relative projection length
    u = np.dot( p - a, d ) / np.dot(d, d)
    u = np.clip(u, 0, 1)

    proj = a + u*d
    return np.linalg.norm(proj - p)

def project(p, a, b):
    ''' Project point p on segment a, b'''
    d = b - a
    t = np.dot( p - a, d ) / np.dot(d, d)
    return a + d*t

def perp(x):
    ''' 2d perpendicular vector'''
    return np.dot([[0,-1],[1, 0]], x)

def reflect( a, b ):
    d = np.dot(b,a)
    return a - b*d*2

def line_intersection_uv( a1, a2, b1, b2, aIsSegment=False, bIsSegment=False):
    EPS = 0.00001
    intersection = np.zeros(2)
    uv = np.zeros(2)

    denom  = (b2[1]-b1[1]) * (a2[0]-a1[0]) - (b2[0]-b1[0]) * (a2[1]-a1[1])
    numera = (b2[0]-b1[0]) * (a1[1]-b1[1]) - (b2[1]-b1[1]) * (a1[0]-b1[0])
    numerb = (a2[0]-a1[0]) * (a1[1]-b1[1]) - (a2[1]-a1[1]) * (a1[0]-b1[0])

    if abs(denom) < EPS:
        return False, intersection, uv

    uv[0] = numera / denom
    uv[1] = numerb / denom

    intersection[0] = a1[0] + uv[0] * (a2[0] - a1[0])
    intersection[1] = a1[1] + uv[0] * (a2[1] - a1[1])

    isa = True
    if aIsSegment and (uv[0]  < 0 or uv[0]  > 1):
        isa = False
    isb = True
    if bIsSegment and (uv[1] < 0 or uv[1]  > 1):
        isb = False

    res = isa and isb
    return res, intersection, uv

def line_intersection( a1, a2, b1, b2, aIsSegment=False, bIsSegment=False ):
    res, intersection, uv = line_intersection_uv(a1,a2,b1,b2,False,False)
    return res, intersection

def line_segment_intersection( a1, a2, b1, b2 ):
    res, intersection, uv = line_intersection_uv(a1,a2,b1,b2,False,True)
    return res, intersection

def segment_line_intersection( a1, a2, b1, b2 ):
    res, intersection, uv = line_intersection_uv(a1,a2,b1,b2,True,False)
    return res, intersection

def segment_intersection( a1, a2, b1, b2 ):
    res, intersection, uv = line_intersection_uv(a1,a2,b1,b2,True,True)
    return res, intersection

def line_ray_intersection( a1, a2, b1, b2 ):
    res, intersection, uv = line_intersection_uv(a1,a2,b1,b2,False,False)
    return res and uv[1] > 0, intersection

def ray_line_intersection( a1, a2, b1, b2 ):
    res, intersection, uv = line_intersection_uv(a1,a2,b1,b2,False,False)
    return res and uv[0] > 0, intersection

def ray_intersection( a1, a2, b1, b2 ):
    res, intersection, uv = line_intersection_uv(a1,a2,b1,b2,False,False)
    return res and uv[0] > 0 and uv[1] > 0, intersection

def ray_segment_intersection( a1, a2, b1, b2 ):
    res, intersection, uv = line_intersection_uv(a1,a2,b1,b2,False,True)
    return res and uv[0] > 0 and uv[1] > 0, intersection

def intersect_lines_lsq(lines, l=0, reg_point=None):
    ''' Least squares line intersection:
        http://cal.cs.illinois.edu/~johannes/research/LS_line_intersect.pdf
        l is a regularization factor, encouraging points toward
        reg_point, if defined, or the midpoint between the segments defining the lines otherwise
    '''

    R = np.zeros((2,2))
    q = np.zeros(2)
    s = np.zeros(2)
    for a, b in lines:
        n = ((b-a) / np.linalg.norm(b-a)).reshape(-1,1)
        #n = np.array([-d[1], d[0]]).reshape(-1,1)
        R += (np.eye(2) - n @ n.T)
        q += (np.eye(2) - n @ n.T) @ a
        s += (a + b)/2
    s = s/len(lines)
    if reg_point is not None:
        s = reg_point
    ins = np.dot(np.linalg.pinv(R + np.eye(2)*l), q + l*s)
    return ins

# Rect utilities
def bounding_box(S, padding=0):
    ''' Axis ligned bounding box of one or more contours (any dimension)
        Returns [min,max] list'''
    if not is_compound(S):
        S = [S]
    if not S:
        return np.array([0, 0]), np.array([0, 0])

    bmin = np.min([np.min(V, axis=0) for V in S ], axis=0)
    bmax = np.max([np.max(V, axis=0) for V in S ], axis=0)
    return [bmin - padding, bmax + padding]

def rect_w(rect):
    return (np.array(rect[1]) - np.array(rect[0]))[0]

def rect_h(rect):
    return (np.array(rect[1]) - np.array(rect[0]))[1]

def rect_size(rect):
    return np.array(rect[1]) - np.array(rect[0])

def rect_aspect(rect):
    return rect_w(rect) / rect_h(rect)

def pad_rect(rect, pad):
    return np.array(rect[0])+pad, np.array(rect[1])-pad

def make_rect(x, y, w, h):
    return [np.array([x, y]), np.array([x+w, y+h])]

def make_centered_rect(p, size):
    return make_rect(p[0] - size[0]*0.5, p[1] - size[1]*0.5, size[0], size[1])

def rect_center(rect):
    return rect[0] + (rect[1]-rect[0])/2

def rect_corners(rect, close=False):
    w, h = rect_size(rect)
    rect = (np.array(rect[0]), np.array(rect[1]))
    P = [rect[0], rect[0] + [w, 0],
            rect[1], rect[0] + [0, h]]
    if close:
        P.append(P[0])
    return P

def rect_l(rect):
    return rect[0][0]

def rect_r(rect):
    return rect[1][0]

def rect_t(rect):
    return rect[0][1]

def rect_b(rect):
    return rect[1][1]
    
def random_point_in_rect(box):
    x = np.random.uniform( box[0][0], box[1][0] )
    y = np.random.uniform( box[0][1], box[1][1] )
    return np.array([x, y])

def scale_rect(rect, s, halign=0, valign=0):
    if is_number(s):
        s = [s, s]
    sx, sy = s
    r = [np.array(rect[0]), np.array(rect[1])]
    origin = rect_center(rect)
    if (halign == -1):
        origin[0] = rect_l(rect)
    if (halign == 1):
        origin[0] = rect_r(rect)
    if (valign == -1):
        origin[1] = rect_t(rect)
    if (valign == 1):
        origin[1] = rect_b(rect)
    A = trans_2d(origin)@scaling_2d([sx, sy])@trans_2d(-origin)

    return [affine_transform(A, r[0]), affine_transform(A, r[1])]


def rect_in_rect(src, dst, padding=0., axis=None):
    ''' Fit src rect into dst rect, preserving aspect ratio of src, with optional padding'''
    dst = pad_rect(dst, padding)

    dst_w, dst_h = dst[1] - dst[0]
    src_w, src_h = src[1] - src[0]

    ratiow = dst_w/src_w
    ratioh = dst_h/src_h

    if axis==None:
        if ratiow <= ratioh:
            axis = 1
        else:
            axis = 0
    if axis==1: # fit vertically [==]
        w = dst_w
        h = src_h*ratiow
        x = dst[0][0]
        y = dst[0][1] + dst_h*0.5 - h*0.5
    else: # fit horizontally [ || ]
        w = src_w*ratioh
        h = dst_h

        y = dst[0][1]
        x = dst[0][0] + dst_w*0.5 - w*0.5

    return make_rect(x, y, w, h)

def rect_to_rect_transform(src, dst):
    ''' Fit src rect into dst rect, without preserving aspect ratio'''
    m = np.eye(3,3)

    sw, sh = rect_size(src)
    dw, dh = rect_size(dst)

    m = trans_2d([dst[0][0],dst[0][1]])
    m = np.dot(m, scaling_2d([dw/sw,dh/sh]))
    m = np.dot(m, trans_2d([-src[0][0],-src[0][1]]))

    return m

def rect_to_rect_transform(src, dst):
    m = np.eye(3,3)

    sw, sh = rect_size(src)
    dw, dh = rect_size(dst)

    m = trans_2d([dst[0][0],dst[0][1]])
    m = np.dot(m, scaling_2d([dw/sw,dh/sh]))
    m = np.dot(m, trans_2d([-src[0][0],-src[0][1]]))

    return m

def rect_in_rect_transform(src, dst, padding=0., axis=None):
    ''' Return homogeneous transformation matrix that fits src rect into dst'''
    fitted = rect_in_rect(src, dst, padding, axis)

    cenp_src = rect_center(src)
    cenp_dst = rect_center(fitted)

    M = np.eye(3)
    M = np.dot(M,
               trans_2d(cenp_dst - cenp_src))
    M = np.dot(M, trans_2d(cenp_src))
    M = np.dot(M, scaling_2d(rect_size(fitted)/rect_size(src)))
    M = np.dot(M, trans_2d(-cenp_src))
    return M

def rect_grid(rect, nrows, ncols, margin=0, padding=0, flatten=True):
    rect = pad_rect(rect, margin)
    w, h = rect_size(rect)
    x0, y0 = rect[0]
    rw = w / (ncols) - padding*2
    rh = h / (nrows) - padding*2
    xs = np.linspace(0, w, ncols+1)[:-1]
    ys = np.linspace(0, h, nrows+1)[:-1]
    rects = []
    for y in ys:
        row = []
        for x in xs:
            row.append(make_rect(x0+x+padding, y0+y+padding, rw, rh))
        rects.append(row)
    if flatten:
        rects = sum(rects, [])
    return rects

def fit_shapes_in_grid(shapes, rect, nrows, ncols, margin=0, padding=0, flatten=True, get_matrices=False, offset=0):
    rects = rect_grid(rect, nrows, ncols, margin, padding)
    fitted = []
    matrices = []
    for i, shape in enumerate(shapes):
        i = i + offset
        if i >= len(rects):
            print('insufficient rows and columns in grid')
            break

        A = rect_in_rect_transform(bounding_box(shape), rects[i])
        shape = affine_transform(A, shape)

        matrices.append(A)

        if flatten:
            fitted += shape
        else:
            fitted.append(shape)
    if get_matrices:
        return fitted, matrices
    return fitted

# 2d transformations (affine)
def rotate_vector_2d(v, ang):
    ''' 2d rotation matrix'''
    ca = np.cos(ang)
    sa = np.sin(ang)
    x, y = v
    return np.array([x*ca - y*sa,
                     x*sa + y*ca])

def rot_2d( theta, affine=True ):
    d = 3 if affine else 2
    m = np.eye(d)
    ct = np.cos(theta)
    st = np.sin(theta)
    m[0,0] = ct; m[0,1] = -st
    m[1,0] = st; m[1,1] = ct

    return m

def trans_2d( xy):
    m = np.eye(3)
    m[0,2] = xy[0]
    m[1,2] = xy[1]
    return m

def scaling_2d( xy, affine=True ):
    d = 3 if affine else 2

    if is_number(xy):
        xy = [xy, xy]

    m = np.eye(d)
    m[0,0] = xy[0]
    m[1,1] = xy[1]
    return m

def shear_2d(xy, affine=True):
    d = 3 if affine else 2
    m = np.eye(d)
    #return m
    m[0,1] = xy[0]
    m[1,0] = xy[1]
    return m

# 3d transformations (affine)
def rotx_3d (theta, affine=True):
    d = 4 if affine else 3
    m = np.eye(d)
    ct = cos(theta)
    st = sin(theta)
    m[1,1] = ct; m[1,2] = -st
    m[2,1] = st; m[2,2] = ct

    return m

def roty_3d (theta, affine=True):
    d = 4 if affine else 3
    m = np.eye(d)
    ct = cos(theta)
    st = sin(theta)
    m[0,0] = ct; m[0,2] = st
    m[2,0] = -st; m[2,2] = ct

    return m

def rotz_3d (theta, affine=True):
    d = 4 if affine else 3
    m = np.eye(d)
    ct = cos(theta)
    st = sin(theta)
    m[0,0] = ct; m[0,1] = -st
    m[1,0] = st; m[1,1] = ct

    return m

def trans_3d(xyz):
    m = np.eye(4)
    m[0,3] = xyz[0]
    m[1,3] = xyz[1]
    m[2,3] = xyz[2]
    return m

def scaling_3d(s, affine=True):
    d = 4 if affine else 4
    if not isinstance(s, (list, tuple, np.ndarray)):
        s = [s, s, s]
    
    m = np.eye(d)
    m[0,0] = s[0]
    m[1,1] = s[1]
    m[2,2] = s[2]
    return m

def parallel(aspect, znear=0.1, zfar=1000):
    w = aspect
    h = 1
    l, r, b, t = -w, w, -h, h
    n, f = znear, zfar
    m = np.eye(4)

    m[0, 0] = 2. / (r - l)
    m[1, 1] = 2. / (t - b)
    m[2, 2] = -2. / (f - n)

    m[0, 3] = -(r + l) / (r - l)
    m[1, 3] = -(t + b) / (t - b)
    m[2, 3] = -(f + n) / (f - n)
    return m

def perspective(fov, aspect, znear=0.1, zfar=1000):
    m = np.zeros((4,4))
    yscale = 1.0 / tan(fov/2)
    xscale = yscale / aspect
    
    m[0,0] = xscale
    m[1,1] = yscale
    
    m[2,2] = zfar / ( znear - zfar )
    m[2,3] = znear*zfar/(znear - zfar )
    
    m[3,2] = -1.0
    return m
    
def frustum( left, right, bottom, top, near, far ):
    m = np.zeros((4,4))

    x = (2.0*near) / (right-left)
    y = (2.0*near) / (top-bottom)
    a = (right+left)/(right-left)
    b = (top+bottom)/(top-bottom)
    c = -(far+near)/(far-near)
    d = -(2.0*far*near)/(far-near)
    
    m[0,0] = x; m[0,1] = 0; m[0,2] = a; m[0,3] = 0
    m[1,0] = 0; m[1,1] = y; m[1,2] = b; m[1,3] = 0
    m[2,0] = 0; m[2,1] = 0; m[2,2] = c; m[2,3] = d
    m[3,0] = 0; m[3,1] = 0; m[3,2] = -1.0; m[3,3] = 0

    return m

def _affine_transform_polyline(mat, P):
    dim = P[0].size
    P = np.vstack([np.array(P).T, np.ones(len(P))])
    P = mat@P
    return list(P[:dim,:].T)

def affine_transform(mat, data):
    if is_empty(data):
        print('Empty data to affine_transform!')
        return data
    if is_polyline(data):
        P = data
        dim = P[0].size
        P = np.vstack([np.array(P).T, np.ones(len(P))])
        P = mat@P
        return P[:dim,:].T
    elif is_compound(data):
        return [affine_transform(mat, P) for P in data]
    else: # assume a point
        dim = len(data)
        p = np.concatenate([data, [1]])
        return (mat@p)[:dim]

def affine_mul(mat, data):
    print('Use affine_transform instead')
    return affine_transform(mat, data) # For backwards compat

def clip_3d(p, q):
    ''' Liang-Barsky homogeneous line clipping to canonical view volume '''
    if p[3] < 0 and q[3] < 0:
        return False, p, q, False, False

    t = [0.0, 1.0]
    d = q - p

    if not clipt(p[3] - p[0], -d[3] + d[0], t): return False, p, q, False, False
    if not clipt(p[3] + p[0], -d[3] - d[0], t): return False, p, q, False, False
    if not clipt(p[3] - p[1], -d[3] + d[1], t): return False, p, q, False, False
    if not clipt(p[3] + p[1], -d[3] - d[1], t): return False, p, q, False, False
    if not clipt(p[3] - p[2], -d[3] + d[2], t): return False, p, q, False, False
    if not clipt(p[3] + p[2], -d[3] - d[2], t): return False, p, q, False, False

    clipq = False
    if t[1] < 1:
        q = p + t[1] * d
        clipq = True
    clipp = False
    if t[0] > 0:
        p = p + t[0] * d
        clipp = True
    return True, p, q, clipp, clipq

def clipt(num, denom, t):
    if denom < 0:
        r = num / denom
        if r > t[1]:
            return False
        if r > t[0]:
            t[0] = r
    elif denom > 0:
        r = num / denom
        if r < t[0]:
            return False
        if r < t[1]:
            t[1] = r
    elif num < 0:
        return False
    return True

def clip_poly_3d(P):
    Pc = [[]]
    def add_segment():
        if Pc[-1]:
            Pc.append([])
    def segment_has_point():
        return Pc[-1]
    def add_point(p):
        Pc[-1].append(p)

    for a, b in zip(P, P[1:]):
        accept, a, b, clipa, clipb = clip_3d(a, b)
        if accept:
            if clipa:
                add_segment()
            if not segment_has_point():
                add_point(a)
            add_point(b)
            if clipb:
                add_segment()
        else:
            add_segment()
    return Pc

def view_3d(polyline, modelview, projection, viewport=[vec(-1,-1), (1,1)], clip=True, get_normalized_coords=False):
    ''' Compute 3D viewing transformation for a list of polylines
    Input:
    polylone: 1 or a list of polylines
    modelview: 4x4 (model)view matrix
    projection: 4x4 projection matrix (use perspective or parallel)
    viewport: view rect
    clip: if True perform clipping to canonical view volume

    Output: projected and possibly clipped 2D polylines
    If clipping is enabled, the number of output polyline is not necessarily the same as the input
    '''

    if is_compound(polyline):
        segments = []
        Z = []
        for pts in polyline:
            if is_empty(pts):
                continue
            all = view_3d(pts, modelview, projection, viewport, clip, get_normalized_coords)
            if get_normalized_coords:
                segments.append(all[0])
                Z.append(all[1])
            else:
                segments.append(all)
        if get_normalized_coords:
            return sum(segments, []), sum(Z, [])
        return sum(segments, [])

    w, h = rect_w(viewport), rect_h(viewport)
    center = rect_center(viewport)
    def to_screen(pw):
        div = 1. / pw[3]
        return vec((pw[0] * div)*0.5*w + center[0],
               (-pw[1] * div)*0.5*h + center[1])

    # model view transform
    polyline = affine_transform(modelview, polyline)
    # to canonical view volume
    pts = np.array(polyline).T
    pts = np.vstack([pts, np.ones(pts.shape[1])])
    Pw = (projection@pts).T
    if clip:
        Pc = clip_poly_3d(Pw)
    else:
        Pc = [Pw]
    Pv = [[to_screen(p) for p in seg] for seg in Pc]
    #pdb.set_trace()
    if get_normalized_coords:
        Z = [[p[:3]/p[3] for p in seg] for seg in Pc]
        return Pv, Z
    return Pv

# Generates shapes (as polylines, 2d and 3d)
class shapes:
    def __init__(self):
        pass

    @staticmethod
    def box_3d(min, max):
        S = []
        # plotter-friendy version
        S.append(shapes.polygon(vec(min[0], min[1], min[2]),
                                vec(max[0], min[1], min[2]),
                                vec(max[0], max[1], min[2]),
                                vec(min[0], max[1], min[2])))
        S.append(shapes.polygon(vec(min[0], min[1], max[2]),
                                vec(max[0], min[1], max[2]),
                                vec(max[0], max[1], max[2]),
                                vec(min[0], max[1], max[2])))
        for i in range(4):
            S.append([S[0][i], S[1][i]])
        # line segments only
        # S.append([vec(min[0], min[1], min[2]),  vec(max[0], min[1], min[2])])
        # S.append([vec(max[0], min[1], min[2]),  vec(max[0], max[1], min[2])])
        # S.append([vec(max[0], max[1], min[2]),  vec(min[0], max[1], min[2])])
        # S.append([vec(min[0], max[1], min[2]),  vec(min[0], min[1], min[2])])
        # S.append([vec(min[0], min[1], max[2]),  vec(max[0], min[1], max[2])])
        # S.append([vec(max[0], min[1], max[2]),  vec(max[0], max[1], max[2])])
        # S.append([vec(max[0], max[1], max[2]),  vec(min[0], max[1], max[2])])
        # S.append([vec(min[0], max[1], max[2]),  vec(min[0], min[1], max[2])])
        # S.append([vec(min[0], min[1], min[2]),  vec(min[0], min[1], max[2])])
        # S.append([vec(min[0], max[1], min[2]),  vec(min[0], max[1], max[2])])
        # S.append([vec(max[0], max[1], min[2]),  vec(max[0], max[1], max[2])])
        # S.append([vec(max[0], min[1], min[2]),  vec(max[0], min[1], max[2])])
        return S

    @staticmethod
    def cuboid(center=vec(0,0,0), halfsize=vec(1,1,1)):
        if is_number(halfsize):
            size = [halfsize, halfsize, halfsize]
        return shapes.box_3d(np.array(center) - np.array(halfsize),
                         np.array(center) + np.array(halfsize))


    @staticmethod
    def polygon(*args):
        ''' A closed polygon (joins last point to first)'''
        P = [np.array(p) for p in args]
        P.append(np.array(args[0]))
        return P

    @staticmethod
    def circle(center, r, subd=80):
        return close_path([vec(np.cos(th), np.sin(th))*r + center
            for th in np.linspace(0, np.pi*2, subd)[:-1]])

    @staticmethod
    def random_radial_polygon(n, min_r=0.5, max_r=1., center=[0,0]):
        import polygonsoup.numeric as numeric
        R = np.random.uniform(min_r, max_r, n)
        start = np.random.uniform(0., np.pi*2)
        Theta = numeric.randspace(start, start+np.pi*2, n+1)
        Theta = Theta[:-1] # skip last one
        V = np.zeros((n,2))
        V[:,0] = np.cos(Theta) * R + center[0]
        V[:,1] = np.sin(Theta) * R + center[1]
        return V

    @staticmethod
    def rectangle(*args):
        if len(args) == 2:
            rect = [*args]
        elif len(args) == 1:
            rect = args[0]
        elif len(args) == 4:
            rect = make_rect(*args)
        P = np.array(rect_corners(rect))
        return P

plane_xy = (vec(1,0,0), vec(0,1,0))
plane_xz = (vec(1,0,0), vec(0,0,1))
plane_zx = (vec(0,0,1), vec(1,0,0))
plane_zy = (vec(0,0,1), vec(0,1,0))

def to_3d_plane(S, basis, origin=vec(0,0,0)):
    if is_compound(S):
        return [to_3d_plane(P, basis, origin) for P in S]
    return [p[0]*basis[0]+p[1]*basis[1]+origin for p in S]

def zup_basis():
    # Z up
    # Y right
    # X fw
    return np.array([
        [0, 0, 1, 0],
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 0, 1]]).T

## Computational geometry utilities

def interior_straight_skeletons(S):
    import skgeom as sg
    polyholes = get_polygons_with_holes(S)
    skels = []
    for P, holes in polyholes:

        poly = sg.PolygonWithHoles(sg.Polygon([sg.Point2(*p) for p in P]),
                                   [sg.Polygon([sg.Point2(*p) for p in Q]) for Q in holes])
        skel = sg.skeleton.create_interior_straight_skeleton(poly)
        skels.append(skel)
    return skels

def _point_to_np(p):
    return np.array([float(p.x()),
                 float(p.y())])

def compute_planar_map(polylines):
    import skgeom
    arr = skgeom.arrangement.Arrangement()

    for P in polylines:
        for a, b in zip(P, P[1:]):
            try:
                arr.insert(skgeom.Segment2(skgeom.Point2(*a),
                                    skgeom.Point2(*b)))
            except Exception as e:
                print(e)

    return arr

def planar_map_faces(planar):
    S = []
    for face in planar.faces:
        face = face_vertices(face)
        if face:
            S.append(close_path(face))
    return S

def find_face(planar, p):
    import skgeom
    face = planar.find(skgeom.Point2(*p))
    return face

def face_vertices(face):
    if face.is_unbounded():
        return []
    P = []
    i = face.outer_ccb
    first = next(i)
    halfedge = next(i)
    while first != halfedge:
        P.append(_point_to_np(halfedge.source().point()))
        halfedge = next(i)
    P.append(_point_to_np(halfedge.source().point()))

    return P

def curvature(P, closed=0):
    ''' Contour curvature'''
    P = P.T
    if closed:
        P = np.c_[P[:,-1], P, P[:,0]]

    D = np.diff(P, axis=1)
    l = np.sqrt(np.sum(np.abs(D)**2,axis=0))+1e-200 #np.sqrt( D[0,:]**2 + D[1,:]**2 )
    D[0,:] /= l
    D[1,:] /= l

    n = D.shape[1] #size(D,2);

    theta = np.array([ angle_between(a, b) for a, b in zip(D.T, D.T[1:]) ])

    K = 2.0*np.sin(theta/2) / (np.sqrt( l[:-1] *l[1:] + 1e-200 ))

    if not closed:
        K = np.concatenate([[K[0]], K, [K[-1]]])

    return K
#endf

def uniform_sample( X, delta_s, closed=0, kind='slinear', data=None, inv_density=None, density_weight=0.5 ):
    ''' Uniformly samples a contour at a step dist'''
    if closed:
        X = np.vstack([X, X[0]])
        if data is not None:
            data = np.vstack([data, data[0]])

    D = np.diff(X[:,:2], axis=0)
    # chord lengths
    s = np.sqrt(D[:,0]**2 + D[:,1]**2)
    # Delete values in input with zero distance (due to a bug in interp1d)
    I = np.where(s==0)
    X = np.delete(X, I, axis=0)
    s = np.delete(s, I)
    # if inv_density is not None:
    #     inv_density = np.delete(inv_density, I)

    if data is not None:
        if type(data)==list or data.ndim==1:
            data = np.delete(data, I)
        else:
            data = np.delete(data, I, axis=0)

    #maxs = np.max(s)
    #s = s/maxs
    #delta_s = delta_s/maxs #np.max(s)

    # if inv_density is not None:
    #     inv_density = inv_density[:-1]
    #     inv_density = inv_density - np.min(inv_density)
    #     inv_density /= np.max(inv_density)
    #     density = density #1.0 - inv_density
    #     s = (1.0 - density_weight)*s + density_weight*density
    u = np.cumsum(np.concatenate([[0.], s]))
    u = u / u[-1]
    n = int(np.ceil(np.sum(s) / delta_s))
    t = np.linspace(u[0], u[-1], n)

    # if inv_density is not None:
    #     inv_density = inv_density - np.min(inv_density)
    #     inv_density /= np.max(inv_density)
    #     inv_density = np.clip(inv_density, 0.75, 1.0)
    #     param = np.cumsum(1-inv_density)
    #     param = param - np.min(param)
    #     param = param / np.max(param)

    #     u = u*param #(1.0 - density_weight) + param*density_weight
    #     u = u/u[-1]

    f = interp1d(u, X.T, kind=kind)
    Y = f(t)

    if data is not None:
        f = interp1d(u, data.T, kind=kind)
        data = f(t)
        if closed:
            if data.ndim>1:
                return Y.T[:-1,:], data.T[:-1,:]
            else:
                return Y.T[:-1,:], data.T[:-1]
        else:
            return Y.T, data.T
    if closed:
        return Y[:,:-1].T
    return Y.T

def smoothing_spline(n, pts, der=0, ds=0., closed=False, w=None, smooth_k=0, degree=3, alpha=1.):
    ''' Computes a smoothing B-spline for a sequence of points.
    Input:
    n, number of interpolating points
    pts, sequence of points (dim X m)
    der, derivative order
    ds, if non-zero an approximate arc length parameterisation is used with distance ds between points,
    and the parameter n is ignored.
    closed, if True spline is periodic
    w, optional weights
    smooth_k, smoothing parameter,
    degree, spline degree,
    alpha, parameterisation (1, uniform, 0.5 centripetal)
    '''
    if closed:
        pts = np.vstack([pts, pts[0]])

    if w is None:
        w = np.ones(pts.shape[0])

    dim = pts.shape[1]
    D = np.diff(pts, axis=0)
    # chord lengths
    s = np.sqrt(np.sum([D[:,i]**2 for i in range(dim)], axis=0))
    I = np.where(s==0)
    pts = np.delete(pts, I, axis=0)
    s = np.delete(s, I)
    w = np.delete(w, I)

    degree = min(degree, pts.shape[0]-1)

    if pts.shape[0] < 2:
        print('Insufficient points for smoothing spline, returning original')
        return pts

    if ds != 0:
        D = np.diff(pts, axis=0)
        s = np.sqrt(np.sum([D[:,i]**2 for i in range(dim)], axis=0))
        l = np.sum(s)
        s = s**(alpha)
        u = np.cumsum(np.concatenate([[0.], s]))
        u = u / u[-1]

        spl, u = splprep(pts.T, w=w, u=u, k=degree, per=closed, s=smooth_k)
        n = max(2, int(l / ds))
        t = np.linspace(u[0], u[-1], n)
    else:
        u = np.linspace(0, 1, pts.shape[0])
        spl, u = splprep(pts.T, u=u, w=w, k=degree, per=closed, s=smooth_k)
        t = np.linspace(0, 1, n)

    if type(der)==list:
        res = []
        for d in der:
            res.append(np.vstack(splev(t, spl, der=d)).T)
        return res
    res = splev(t, spl, der=der)
    return np.vstack(res).T

def cleanup_contour(X, closed=False, eps=1e-10, get_inds=False):
    ''' Removes points that are closer then a threshold eps'''
    if closed:
        X = np.vstack([X, X[0]])
    D = np.diff(X, axis=0)
    inds = np.array(range(X.shape[0])).astype(int)
    # chord lengths
    s = np.sqrt(D[:,0]**2 + D[:,1]**2)
    # Delete values in input with zero distance (due to a bug in interp1d)
    I = np.where(s<eps)[0]

    if closed:
        X = X[:-1]
    if len(I):
        X = np.delete(X, I, axis=0)
        inds = np.delete(inds, I)
    if get_inds:
        return X, inds
    return X


def chord_lengths( P, closed=0 ):
    ''' Chord lengths for each segment of a contour '''
    if closed:
        P = np.vstack([P, P[0]])
    D = np.diff(P, axis=0)
    L = np.sqrt( D[:,0]**2 + D[:,1]**2 )
    return L


def cum_chord_lengths( P, closed=0 ):
    ''' Cumulative chord lengths '''
    if len(P.shape)!=2:
        return []
    if P.shape[0] == 1:
        return np.zeros(1)
    L = chord_lengths(P, closed)
    return np.cumsum(np.concatenate([[0.0],L]))

def chord_length( P, closed=0 ):
    ''' Chord length of a contour '''
    if len(P.shape)!=2 or P.shape[0] < 2:
        return 0.
    L = chord_lengths(P, closed)
    return np.sum(L)

def polygon_area(P):
    if len(P.shape) < 2 or P.shape[0] < 3:
        return 0
    n = P.shape[0]
    area = 0.0
    for i in range(n):
        p0 = i
        p1 = (i+1)%n
        area += P[p0,0] * P[p1,1] - P[p1,0] * P[p0,1]
    return area * 0.5

def triangle_area( a, b, c ):
    da = a-b
    db = c-b
    return det(np.vstack([da, db]))*0.5

def left_of(p, a, b, eps=1e-10):
    # Assumes coordinate system with y up so actually will be "right of" if visualizd y down
    p, a, b = [np.array(v) for v in [p, a, b]]
    return triangle_area(a, b, p) < eps

def is_point_in_triangle(p, tri, eps=1e-10):
    L = [left_of(p, tri[i], tri[(i+1)%3], eps) for i in range(3)]
    return L[0]==L[1] and L[1]==L[2]

def is_point_in_rect(p, rect):
    ''' return wether a point is in a rect'''
    l, t = rect[0]
    r, b = rect[1]
    w, h = rect[1] - rect[0]

    return ( p[0] >= l and p[1] >= t and
             p[0] <= r and p[1] <= b )

def is_point_in_poly(p, P):
    ''' Return true if point in polygon'''
    c = False
    n = P.shape[0]
    j = n-1
    for i in range(n):
        if ( ((P[i,1]>p[1]) != (P[j,1]>p[1])) and
             (p[0] < (P[j,0] - P[i,0])*(p[1] - P[i,1]) / (P[j,1] - P[i,1]) + P[i,0]) ):
                 c = not c
        j = i
    return c
   
def is_point_in_shape(p, S, get_flags=False):
    ''' Even odd point in shape'''
    c = 0
    flags = []
    for P in S:
        if is_point_in_poly(p, P):
            flags.append(True)
            c = c+1
        else:
            flags.append(False)

    res = (c%2) == 1
    if get_flags:
        return res, flags
    return res

# Circles
def circular_segment_area( r, h ):
    return r*r*np.acos((r-h)/r)-(r-h)*np.sqrt(2.0*r*h - h*h)

def circumcircle_radius( pa, pb, pc ):
    d = distance(pa,pb) * distance(pb,pc) * distance(pc,pa)
    d /= (2.0 * np.abs(triangle_area(pa, pb, pc)) + 1e-10)
    return d / 2

# code adapted from http://www.ics.uci.edu/~eppstein/junkyard/circumcenter.html
def circumcenter(a, b, c, exact=True):
    def orient2d( pa, pb, pc ):
        acx = pa[0] - pc[0]
        bcx = pb[0] - pc[0]
        acy = pa[1] - pc[1]
        bcy = pb[1] - pc[1]
        return acx * bcy - acy * bcx

    # Use coordinates relative to point `a' of the triangle.
    xba = b[0] - a[0]
    yba = b[1] - a[1]
    xca = c[0] - a[0]
    yca = c[1] - a[1]

    # Squares of lengths of the edges incident to `a'.
    balength = xba * xba + yba * yba
    calength = xca * xca + yca * yca

    # Calculate the denominator of the formulae.
    if exact:
        # Use orient2d() from http://www.cs.cmu.edu/~quake/robust.html
        # to ensure a correctly signed (and reasonably accurate) result,
        # avoiding any possibility of division by zero.
        denominator = 0.5 / orient2d(b, c, a)
    else:
        # Take your chances with floating-point roundoff.
        denominator = 0.5 / (xba * yca - yba * xca);

    # Calculate offset (from `a') of circumcenter.
    xcirca = (yca * balength - yba * calength) * denominator
    ycirca = (xba * calength - xca * balength) * denominator

    return np.array([a[0]+xcirca, a[1]+ycirca])

def orthocenter(a, b, c):
    pa = project(a, b, c)
    pb = project(b, a, c)
    return line_intersection(a, pa, b, pb)[1]

def circumcircle(a, b, c):
    cr = circumcircle_radius(a, b, c)
    cp = circumcenter(a, b, c)
    return cp, cr

def incircle(pa, pb, pc):
    ''' Incircle of a triangle, returns (center, radius)
    # https://people.sc.fsu.edu/~jburkardt/py_src/triangle_properties/triangle_incircle.py
    '''
    a = np.sqrt ( ( pa[0] - pb[0] ) ** 2 + ( pa[1] - pb[1] ) ** 2 )
    b = np.sqrt ( ( pb[0] - pc[0] ) ** 2 + ( pb[1] - pc[1] ) ** 2 )
    c = np.sqrt ( ( pc[0] - pa[0] ) ** 2 + ( pc[1] - pa[1] ) ** 2 )

    perimeter = a + b + c
    center = np.zeros(2)

    if perimeter == 0.0:
        center[0] = pa[0]
        center[1] = pa[1]
        r = 0.0
        return r, center

    center[0] = (  \
          b * pa[0] \
        + c * pb[0] \
        + a * pc[0] ) / perimeter

    center[1] = (  \
          b * pa[1] \
        + c * pb[1] \
        + a * pc[1] ) / perimeter

    r = 0.5 * np.sqrt ( \
          ( - a + b + c ) \
        * ( + a - b + c ) \
        * ( + a + b - c ) / perimeter )

    return center, r

def incenter(pa, pb, pc):
    return incircle(pa, pb, pc)[0]

def circle_overlap(c1, r1, c2, r2):
    ''' Area based measure [0,1] of circle overlap'''
    A = circle_intersection_area(c1, r1, c2, r2)
    amin = min( np.pi * r1**2, np.pi * r2**2 )
    return A / amin

def circles_intersect(c1, r1, c2, r2, eps=0.):
    ''' Returns if two circles intersect,
    returns 2 if circles fully overlap,
    eps defines a proportional threshold on the circle with largest radius'''
    tol = 1. + eps
    if r1 > r2:
        r1 *= tol
    else:
        r2 *= tol
    d = np.linalg.norm(c2 - c1)
    ins = d < (r1+r2)
    if ins:
        if d <= abs(r1-r2):
            return 2
        return 1
    return 0

def circle_intersection_angle(c1, r1, c2, r2):
    ''' Returns the intersection angle (in degrees) between two circles,
        with no intersection, angle is 0
        with full (containement) angle is 180'''
    d = np.linalg.norm(c2 - c1)
    ins = d < (r1+r2)
    if ins:
        if d <= abs(r1-r2):
            return 180.
        else:
            return degrees(np.arccos((d**2 - r1**2 - r2**2) / (2.*r1*r2)))
    else:
        return 0.

def circle_intersection_area(c1, r1, c2, r2):
    ''' Returns area of the intersection of two circles'''
    def A():
        # From http://mathworld.wolfram.com/Circle-CircleIntersection.html
        d = np.linalg.norm(c1 - c2)
        R = r1
        r = r2
        if d >= r + R:
            return 0.

        def fA(R, d):
            return (R**2 * np.arccos(d/R) - d * np.sqrt(R**2 - d**2))

        d1 = (d**2 - r**2 + R**2) / (d*2)
        d2 = d - d1

        A = fA(R,d1) + fA(r,d2)
        return A

    d = np.linalg.norm(c2 - c1)
    ins = d < (r1+r2)
    if ins:
        if d <= abs(r1-r2):
            return min( np.pi * r1**2, np.pi * r2**2 )
        return A()
    return 0.

def circle_union_area(c1, r1, c2, r2):
    ''' Returns area of the union of two circles'''
    AiB = circle_intersection_area(c1, r1, c2, r2)
    A = np.pi * r1**2
    B = np.pi * r2**2
    return A + B - AiB

def select_convex_vertex(P, area=None):
    ''' Select convex vertex (with arbitrary winding)'''
    if area is None:
        area = polygon_area(P)
    n = len(P)
    maxh = 0
    verts = []
    for v in range(n):
        a, b = (v-1)%n, (v+1)%n
        if angle_between(P[v] - P[a], P[b] - P[v])*area > 0:
            h = point_line_distance(P[v], P[a], P[b])
            if h > maxh:
                maxh = h
                verts.append(v)
    return verts[-1]

def get_point_in_polygon(P, area=None):
    ''' Get a point inside polygon P
        See http://apodeline.free.fr/FAQ/CGAFAQ/CGAFAQ-3.html
        and O'Rourke'''
    n = len(P)
    if area is None:
        area = polygon_area(P)
    v = select_convex_vertex(P, area)
    a, b = (v-1)%n, (v+1)%n
    inside = []
    dist = np.inf
    # Check if no other point is inside the triangle a, v, b
    # and select closest to v if any present
    for i in range(n-3):
        q = (b+1+i)%n
        if is_point_in_triangle(P[q], [P[a], P[v], P[b]]):
            d = distance(P[q], P[v])
            if d < dist:
                dist = d
                inside.append(q)
    if not inside:
        return (P[a] + P[b])/2
    # no points inside triangle, select midpoint
    return (P[inside[-1]] + P[v])/2

def get_holes(S, get_points_and_areas=False):
    '''Return an array with same size as S with 0 not a hole an 1 a hole
    Optionally return positions in sub-contours and their areas'''
    areas = [polygon_area(P) for P in S]
    points = [get_point_in_polygon(P, area) for P, area in zip(S, areas)]
    holes = [True if not is_point_in_shape(points[i], S) else False  for i, P in enumerate(S)]
    if get_points_and_areas:
        return holes, points, areas
    return holes


def select_convex_vertex(P, area=None):
    ''' Select convex vertex (with arbitrary winding)'''
    if area is None:
        area = polygon_area(P)
    n = len(P)
    maxh = 0
    verts = []
    for v in range(n):
        a, b = (v-1)%n, (v+1)%n
        if angle_between(P[v] - P[a], P[b] - P[v])*area > 0:
            h = point_line_distance(P[v], P[a], P[b])
            if h > maxh:
                maxh = h
                verts.append(v)
    return verts[-1]


def get_point_in_polygon(P, area=None):
    ''' Get a point inside polygon P
        See http://apodeline.free.fr/FAQ/CGAFAQ/CGAFAQ-3.html
        and O'Rourke'''
    n = len(P)
    if area is None:
        area = polygon_area(P)
    v = select_convex_vertex(P, area)
    a, b = (v-1)%n, (v+1)%n
    inside = []
    dist = np.inf
    # Check if no other point is inside the triangle a, v, b
    # and select closest to v if any present
    for i in range(n-3):
        q = (b+1+i)%n
        if is_point_in_triangle(P[q], [P[a], P[v], P[b]]):
            d = distance(P[q], P[v])
            if d < dist:
                dist = d
                inside.append(q)
    if not inside:
        return (P[a] + P[b])/2
    # no points inside triangle, select midpoint
    return (P[inside[-1]] + P[v])/2


def get_holes(S, get_points_and_areas=False):
    '''Return an array with same size as S with 0 not a hole an 1 a hole
    Optionally return positions in sub-contours and their areas'''
    areas = [polygon_area(P) for P in S]
    points = [get_point_in_polygon((S, i), area) for i, area in enumerate(areas)]
    holes = [True if not is_point_in_shape(points[i], S) else False  for i, P in enumerate(S)]
    if get_points_and_areas:
        return holes, points, areas
    return holes

def get_points_in_holes(S):
    '''Get positions inside the holes of S (if any)'''
    holes, points, areas = get_holes(S, get_points_and_areas=True)
    return [points[i] for i, hole in enumerate(holes) if hole]

def fix_shape_winding(S):
    ''' Fixes shape winding to be consistent:
    for y-down: cw out, ccw in
    for y-up: ccw out, cw in'''
    is_shape = True
    if type(S) != list:
        S = [S]
        is_shape = False
    # Make sure that contours don't have repeated end-points because that would break
    # subsequent computations
    S = [geom.cleanup_contour(P, closed=True) for P in S]
    # Identify holes

    holes, points, areas = get_holes(S, get_points_and_areas=True)
    S2 = []

    for i, P in enumerate(S):
        P = np.array(P)
        if abs(areas[i]) < 1e-10:
            print("zero area sub-shape")
            continue
        if (areas[i] < 0) != holes[i]:
            P = P[::-1]
        S2.append(P)

    if not is_shape:
        return S2[0]
    return S2

def get_polygons_with_holes(S):
    '''Return an array with same size as S with 0 not a hole an 1 a hole
    Optionally return positions in sub-contours and their areas'''
    import pdb
    areas = [polygon_area(P) for P in S]
    points = [get_point_in_polygon(P, area) for P, area in zip(S, areas)]
    holes = []
    points_in_flags = []
    for i, P in enumerate(S):
        res, flags = is_point_in_shape(points[i], S, get_flags=True)
        holes.append(not res)
        points_in_flags.append(flags)

    n = len(S)
    polyholes = []

    for i in range(n):
        if holes[i]:
            continue
        P = S[i]
        pholes = []
        for j in range(n):
            if i==j or not holes[j]:
                continue
            if points_in_flags[j][i]:
                pholes.append(S[j])
        polyholes.append((P, pholes))
    return polyholes

#%%
if __name__=='__main__':
    from importlib import reload
    def test_point_in_polygon():
        from polygonsoup import plut, geom
        reload(geom)
        plut.figure((4,4))
        P = geom.shapes.random_radial_polygon(10, 0.1, 1)
        plut.stroke(P, 'k', closed=True)
        v = geom.select_convex_vertex(P, geom.polygon_area(P))
        plut.fill_circle(P[v], 0.02, 'r')
        p = geom.get_point_in_polygon(P)
        plut.fill_circle(p, 0.02, 'b')
        plut.show()

    def test_fit_shapes():
        import polygonsoup.plut as plut
        reload(plut)
        plut.figure()
        s = [[shapes.random_radial_polygon(7)] for i in range(8)]
        fitted = fit_shapes_in_grid(s, make_rect(0, 0, 100, 100), 3, 3, 10, 4)
        plut.stroke(close(fitted))
        plut.show()

    def test_get_holes():
        from polygonsoup import plut, geom
        reload(geom)
        reload(plut)
        plut.figure((4,4))
        P = geom.shapes.random_radial_polygon(10, 0.5, 1)*2
        Q = geom.shapes.random_radial_polygon(10, 0.5, 1)*0.5
        R = geom.shapes.random_radial_polygon(10, 0.5, 1, center=[3,0])
        S = [P, Q, R]
        S = geom.fix_shape_winding(S)
        #plut.stroke(S, 'k', closed=True)
        holes = geom.get_holes(S)
        for i, P in enumerate(S):
            print(geom.polygon_area(P))
            plut.draw_arrow(P[0], P[1], 'r', head_width=0.01)
            if holes[i]:
                plut.stroke(P, 'r', closed=True)
            else:
                plut.stroke(P, 'k', closed=True)
        for p in geom.get_points_in_holes(S):
            plut.fill_circle(p, 0.05, 'r')
        plut.show()

    #test_fit_shapes()
    #test_point_in_polygon()
    test_get_holes()
