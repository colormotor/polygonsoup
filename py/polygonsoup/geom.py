#%%
#!/usr/bin/env python3
import copy
import numpy as np
from numpy import (sin, cos, tan)
from numpy.linalg import (norm, det, inv)
from scipy.interpolate import interp1d
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

def fit_shapes_in_grid(shapes, rect, nrows, ncols, margin=0, padding=0, flatten=True):
    rects = rect_grid(rect, nrows, ncols, margin, padding)
    fitted = []
    for i, shape in enumerate(shapes):
        if i >= len(rects):
            print('insufficient rows and columns in grid')
            break
        A = rect_in_rect_transform(bounding_box(shape), rects[i])
        shape = affine_transform(A, shape)
        if flatten:
            fitted += shape
        else:
            fitted.append(shape)
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
        return Y[:-1,:].T
    return Y.T

def cleanup_contour(X, eps=1e-10, get_inds=False):
    ''' Removes points that are closer then a threshold eps'''
    D = np.diff(X, axis=0)
    inds = np.array(range(X.shape[0])).astype(int)
    # chord lengths
    s = np.sqrt(D[:,0]**2 + D[:,1]**2)
    # Delete values in input with zero distance (due to a bug in interp1d)
    I = np.where(s<eps)[0]

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

if __name__=='__main__':
    from importlib import reload
    def test_fit_shapes():
        import polygonsoup.plut as plut
        reload(plut)
        plut.figure()
        s = [[shapes.random_radial_polygon(7)] for i in range(8)]
        fitted = fit_shapes_in_grid(s, make_rect(0, 0, 100, 100), 3, 3, 10, 4)
        plut.stroke(close(fitted))
        plut.show()

    test_fit_shapes()
