#!/usr/bin/env python3
#%%
import numpy as np
import polygonsoup.geom as geom
import polygonsoup.plot as plot
import polygonsoup.plotters as plotters
from polygonsoup.geom import (radians, degrees,
                              trans_3d, rotx_3d, roty_3d, rotz_3d)

p = np.zeros(3) # initial position
a = np.pi/2 # Angle (play with this)

steps = [(geom.rotx_3d(a, affine=False), 1), # Rotation, and distance to travel
         (geom.rotx_3d(-a, affine=False), 1),
         (geom.roty_3d(a, affine=False), 1),
         ]

P = [p] # First point in path
H = np.eye(3) # Initial coordinate frame for turtle

for i in range(3): # Try repeating the steps multiple times
    for R, d in steps:
        H = H @ R # Update frame
        p = p + H[:3,2]*d # update position (using Z as "forward")
        P.append(p)

S = [P] # The 3d polylines that will be transformed (in this case 1)

# axes (for reference)
axes = [np.array([[0, 0, 0], [1, 0, 0]]),
        np.array([[0, 0, 0], [0, 1, 0]]),
        np.array([[0, 0, 0], [0, 0, 1]])]

# Camera transformation (Y points up by default)
view = (trans_3d([0, 0, -10.5]) @
        rotx_3d(0.5) @ # X-rotation (so we view a bit from top)
        roty_3d(0.3)) # Y-rotation

# View rect
viewport = geom.make_rect(0, 0, 4, 4)
# Projection matrix (switch between perspective and parallel)
#proj = geom.perspective(geom.radians(30), geom.rect_aspect(viewport))
proj = geom.parallel(geom.rect_aspect(viewport))

# Viewport transformations 3d -> 2d
Sv = geom.view_3d(S, view, proj, viewport, clip=False) # Polylines
axes_v = geom.view_3d(axes, view, proj, viewport, clip=False) # Axes (for reference)

plot.figure('A5')
# draw axes
plot.stroke(axes_v[0], 'r') # X
plot.stroke(axes_v[1], 'g') # Y
plot.stroke(axes_v[2], 'b') # Z

plot.stroke(Sv, 'k', linewidth=2.) # Path
plot.show()
