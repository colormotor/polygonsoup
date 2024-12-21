'''
  _   _   _   _   _   _   _   _   _   _   _
 / \ / \ / \ / \ / \ / \ / \ / \ / \ / \ / \
( P | O | L | Y | G | O | N | S | O | U | P )
 \_/ \_/ \_/ \_/ \_/ \_/ \_/ \_/ \_/ \_/ \_/

Plotter-friendly graphics utilities
© Daniel Berio (@colormotor) 2021 - ...

skeletal_strokes - skeletal stroke implementation(s)
Warps a "prototype" shape along a spine resulting in a deformed "flesh"
Hsu (1984) Skeletal Strokes
'''

import polygonsoup.geom as geom
import numpy as np
from scipy.interpolate import splprep, splev

def curved_skeletal_stroke(prototype, spine_, widths, closed=False, smooth_k=0, degree=3, xfunc=lambda t: t):
    '''Simplified warping along a spine assumed to be a relatively smooth curve'''
    # Avoid coincident points
    spine, I = geom.cleanup_contour(spine_, get_inds=True)
    widths = np.array(widths)[I]

    # smoothing spline
    n = spine.shape[0]
    degree = min(degree, n-1)
    # parameterized py (approximate) arc lenth
    u = geom.cum_chord_lengths(spine)
    u = u/u[-1]
    spl, u = splprep(np.vstack([spine.T, widths]), u=u, k=degree, per=closed, s=smooth_k)
    x, y, w = splev(u, spl)
    dx, dy, dw = splev(u, spl, der=1)

    # normalize prototype
    box = geom.bounding_box(prototype)
    w, h = geom.rect_size(box)
    prototype = geom.affine_transform( geom.scaling_2d([1/w, 1/h])@geom.trans_2d(-box[0] - [0, h/2]), prototype )

    # warp
    flesh = []
    for P in prototype:
        t = xfunc(P[:,0])
        h = P[:,1]
        x, y, w = splev(t, spl)
        dx, dy, dw = splev(t, spl, der=1)
        centers = np.vstack([x, y])
        tangents = np.vstack([dx, dy]) / np.sqrt(dx**2 + dy**2)
        normals = np.vstack([-tangents[1,:], tangents[0,:]])
        Q = centers + normals*h*w
        flesh.append(Q.T)
    return flesh

def random_stroke(spine, wmin, wmax, n=0, degree=3, closed=False, smooth_k=0):
    # smoothing spline
    if n==0:
        n = spine.shape[0]
    #print(closed)
    degree = min(degree, n-1)
    # parameterization
    u = np.linspace(0, 1, spine.shape[0]) #geom.cum_chord_lengths(spine)
    u = u/u[-1]
    spl, u = splprep(spine.T, u=u, k=degree, per=closed, s=smooth_k)
    t = np.linspace(0, 1, n)
    x, y = splev(t, spl)
    dx, dy = splev(t, spl, der=1)

    # if len(spine) > 2:
    #     ddx, ddy = splev(t, spl, der=2)
    #     K =  abs((dx * ddy - dy * ddx) / np.power(dx**2 + dy**2, 3./2))
    # else:
    #     K = np.zeros(len(spine))
    K = np.random.uniform(wmin, wmax) # K / (np.max(K) + 1e-3)
    w = wmin + (wmax-wmin)*K
    centers = np.vstack([x, y])
    tangents = np.vstack([dx, dy]) / np.sqrt(dx**2 + dy**2)
    normals = np.vstack([-tangents[1,:], tangents[0,:]])

    return np.vstack([(centers + normals*w).T,
                      (centers - normals*w).T[::-1]])
    pts = (centers + normals*w).T
    if closed:
        pts = np.vstack([pts, pts[0]])

    #return centers.T
    res = geom.smoothing_spline(n, pts, smooth_k=smooth_k, closed=closed)
    if closed:
        res = np.vstack([res, res[0]])
    return res

def curved_offset(spine, widths, n=0, degree=3, closed=False, smooth_k=0):
    # smoothing spline
    if n==0:
        n = spine.shape[0]
    #print(closed)
    degree = min(degree, n-1)
    # parameterization
    u = np.linspace(0, 1, spine.shape[0]) #geom.cum_chord_lengths(spine)
    u = u/u[-1]
    spl, u = splprep(np.vstack([spine.T, widths]), u=u, k=degree, per=closed, s=smooth_k)
    t = np.linspace(0, 1, n)
    x, y, w = splev(t, spl)
    dx, dy, dw = splev(t, spl, der=1)

    centers = np.vstack([x, y])
    tangents = np.vstack([dx, dy]) / (np.sqrt(dx**2 + dy**2) + 1e-10)
    normals = np.vstack([-tangents[1,:], tangents[0,:]])

    pts = (centers + normals*w).T
    if closed:
        pts = np.vstack([pts, pts[0]])

    #print(pts)

    #return centers.T
    res = geom.smoothing_spline(n, pts, smooth_k=smooth_k, closed=closed)
    if closed:
        res = np.vstack([res, res[0]])

    return res


def fat_path(P, W, closed=False, miter_limit=2, angle_thresh=160):
    W = np.array(W)
    if len(P) < 2:
        return []
    if closed:
        #P = np.vstack([P, P[0]])
        W = np.concatenate([W, [W[-1]]])
        #closed = False
    D = geom.tangents(P, closed)
    N = [-geom.perp(geom.normalize(d)) for d in D]
    Alpha = geom.turning_angles(P, closed, True)
    I = np.where(np.abs(Alpha) > geom.radians(angle_thresh))[0]
    if len(I):
        # print((len(P), len(W)))
        I = [0] + list(I) + [len(P)-1]
        # print([[a, b] for a, b in zip(I, I[1:])])
        #print(I)
        return sum([fat_path(P[a:b+1], W[a:b], False, miter_limit) for a, b in zip(I, I[1:])], [])

    if W.ndim < 2:
        W = np.vstack([W, W]).T

    for i in I:
        plut.fill_circle(P[i], 0.5, 'r')
    m = len(D)
    frames = []
    frame_count = m if closed else m - 1

    # local coordinate frames
    for i in range(frame_count):
        p = P[(i + 1)%m]
        d1 = geom.normalize(D[i])
        d2 = geom.normalize(D[(i + 1)%m])
        w1 = W[i, 1]
        w2 = W[(i+1)%m, 0]
        alpha = Alpha[(i + 1)%m]
        if abs(alpha) < 1e-5:
            alpha = 1e-5
        o1 = w2 / np.sin(alpha);  # eq 2
        o2 = w1 / np.sin(alpha);  # eq 2
        u1 = d1 * o1*np.sign(alpha)
        u2 = -d2 * o2*np.sign(alpha)
        frames.append((u1, u2))

    # envelope
    L = [P[0] + N[0]*W[0,0]]
    R = [P[0] - N[0]*W[0,0]]

    for i in range(frame_count):
        p    = P[(i + 1) % m];
        u1o1 = frames[i][0];
        u2o2 = frames[i][1];

        alpha = Alpha[(i + 1) % m]
        b = u1o1 + u2o2
        #plut.draw_line(p, p+u1o1, 'r')
        #plut.draw_line(p, p+u2o2, 'b')
        #plut.draw_line(p, p-b, 'm')

        unfold = True
        ip1 = (i + 1) % m
        limit = max(W[i,1], W[ip1,0]) * miter_limit

        d1    = D[i]
        d2    = D[ip1]
        hu1   = geom.normalize(u1o1)
        hu2   = geom.normalize(u2o2)

        if alpha < 0.:
            concave_side = L
            convex_side  = R
        else:
            concave_side = R
            convex_side  = L

        bb = [b, b] # apply_miter(b, p, d1, d2, limit)
        convex_side.append(p + bb[0])
        convex_side.append(p + bb[1])
        #concave_side.append(p - b + hu1)
        #concave_side.append(p - b + hu2)
        concave_side.append(p - b)
        concave_side.append(p - b)

    alpha = Alpha[0]
    if alpha < 0.:
        concave_side = L
        convex_side  = R
    else:
        concave_side = R
        convex_side  = L

    if not closed:
        L.append(P[-1] + N[-1] * W[-1, 1])
        R.append(P[-1] - N[-1] * W[-1, 1])
    else:
        L.append(L[0])
        R.append(R[0])
        #concave_side[0] = concave_side[-1]
        #convex_side[0]  = convex_side[-1]
        #concave_side.pop()
        #convex_side.pop()
    #plut.stroke(np.array(L), 'r')
    #plut.stroke(np.array(R), 'b')
    envelope = np.array(L + R[::-1])
    return [envelope]

def apply_miter(b, p, d1, d2, limit):
    l = np.linalg.norm(b)
    if l <= limit:
        return [b, b]
    bu = b / l
    bp = geom.perp(bu) * 10

    p1a = p + b
    p1b = p1a - d1

    p2a = p + b
    p2b = p2a + d2

    bp1 = np.zeros(2)
    bp2 = np.zeros(2)
    res, bp1 = geom.line_segment_intersection(
                            p + bu * limit,
                            p + bu * limit + bp,
                            p1a,
                            p1b)
    res, bp2 = geom.line_segment_intersection(
                            p + bu * limit,
                            p + bu * limit + bp,
                            p2a,
                            p2b)
    return [bp1 - p, bp2 - p]
