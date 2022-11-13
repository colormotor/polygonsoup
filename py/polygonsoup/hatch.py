'''
  _   _   _   _   _   _   _   _   _   _   _
 / \ / \ / \ / \ / \ / \ / \ / \ / \ / \ / \
( P | O | L | Y | G | O | N | S | O | U | P )
 \_/ \_/ \_/ \_/ \_/ \_/ \_/ \_/ \_/ \_/ \_/

Plotter-friendly graphics utilities
© Daniel Berio (@colormotor) 2021 - ...

hatch - 2d hatching
'''

import numpy as np
from collections import namedtuple
from polygonsoup.geom import (radians,
                           affine_transform,
                           rot_2d,
                            bounding_box,
                            is_compound)

Edge = namedtuple('Edge', 'a b m i')

def hatch(S, dist, angle=0., flip_horizontal=False, get_hole_count=False, max_count=1000000, eps=1e-10):
    """Generate scanlines for a possibly compound shape

    Args:
        S (list of or nd.array): contours stored as point lists
        dist (float): distance between scanlines
        angle (float, optional): orientation of scanlines (degrees) . Defaults to 0..
        flip_horizontal (bool, optional): if True, alternates direction of scanlines (for plotting). Defaults to False.
        max_count (int, optional): maximum number of scanlines. Defaults to 1000000.
    
    Returns:
        list of tuples: scanline segments 
    """    
    if not is_compound(S):
        S = [S]

    hole_count = [0 for i in range(len(S))]
    solid_count = [0 for i in range(len(S))]

    if not S:
        return []
    
    # Rotate shape for oriented hatches 
    theta = radians(angle)
    mat = rot_2d(-theta, affine=True)
    S = [affine_transform(mat, P) for P in S]

    box = bounding_box(S)

    # build edge table
    ET = []
    for i, P in enumerate(S):
        P = np.array(P)
        n = P.shape[0]
        if n <= 2:
            continue
        for j in range(n):
            a, b = P[j], P[(j+1)%n]
            # reorder increasing y
            if a[1] > b[1]:
                a, b = b, a
            # slope
            dx = (b[0] - a[0])  
            dy = (b[1] - a[1])
            if abs(dx) > eps:
                m = dy/dx                
            else:
                m = 1e15
            if abs(m) < eps:
                m = None
            ET.append(Edge(a=a, b=b, m=m, i=i))

    # sort by increasing y of first point
    ET = sorted(ET, key=lambda e: e.a[1])

    # intersection x
    def ex(e, y):
        if e.m is None:
            return None
        return e.a[0] + (y - e.a[1])/e.m

    y = box[0][1]
    scanlines = []

    AET = [] # active edge table

    flip = 0
    c = 0
    while ET or AET:
        if y > box[1][1]:
            break
        if c >= max_count:
            print("scanlines: reached max number of iterations")
            break
        c += 1

        # move from ET to AET
        i = 0
        for e in ET:
            if e.a[1] <= y:
                AET.append(e)
                i += 1
            else:
                break
        if i < len(ET):
            ET = ET[i:]
        else:
            ET = []
        
        # remove passed edges
        AET = sorted(AET, key=lambda e: e.b[1])
        AET = [e for e in AET if e.b[1] > y]  
        
        xs = [(ex(e, y), e.i) for e in AET]
        #brk()
        xs = [xi for xi in xs if xi[0] is not None]
        # sort Xs (flipped each scanline for more efficent plotting )
        if flip:
            xs = sorted(xs, key=lambda v: -v[0])
        else:
            xs = sorted(xs, key=lambda v: v[0])
            
        if flip_horizontal:
            flip = not flip
        
        even_odd = [0 for i in range(len(S))]

        if len(xs) > 1:
            #brk()
            parity = 1
            for (x1,i1), (x2,i2) in zip(xs, xs[1:]):   
                a, b = (np.array([x1, y]),
                        np.array([x2, y]))
                if parity:
                    scanlines += [a, b]
                    even_odd[i2] += 1
                else:
                    # If se are outside of a shape and we enounter 
                    # an unvisited contour, it means that this is a separate 
                    # outer contour, so don't count. Otherwise...
                    if even_odd[i2]:
                        even_odd[i2] += 1
                    pass
                parity = not parity

        # increment
        y = y + dist

    # unrotate
    if scanlines:
        scanlines = affine_transform(mat.T, scanlines) #np.array(scanlines))
        # make list of hatch segments
        scanlines = [np.array([a, b]) for a, b in zip(scanlines[0::2], scanlines[1::2])]
    return scanlines
