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

def curved_skeletal_stroke(prototype, spine_, widths, closed=False, smooth_k=10, degree=3):
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
    prototype = geom.affine_transform( geom.scaling_2d([1.0/w, 1.0/w])@geom.trans_2d(-box[0] - [0, h/2]), prototype )

    # warp
    flesh = []
    for P in prototype:
        t = P[:,0]
        h = P[:,1]
        x, y, w = splev(t, spl)
        dx, dy, dw = splev(t, spl, der=1)
        centers = np.vstack([x, y])
        tangents = np.vstack([dx, dy]) / np.sqrt(dx**2 + dy**2)
        normals = np.vstack([-tangents[1,:], tangents[0,:]])
        Q = centers + normals*h*w
        flesh.append(Q.T)
    return flesh
