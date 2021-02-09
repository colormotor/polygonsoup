#!/usr/bin/env python3
# Demonstrates basic 3d view transformations and clipping
#%%

from importlib import reload
import numpy as np
import polygonsoup.geom as geom
import polygonsoup.plot as plot
import polygonsoup.plotters as plotters
reload(geom); reload(plot)
from polygonsoup.geom import (vec,
                              make_rect, rect_aspect,
                              shapes,
                              radians, degrees,
                              trans_3d, rotx_3d, roty_3d,
                              affine_transform, perspective, view_3d)

# Viewport rect
viewport = make_rect(0, 0, 400, 200)

# Create a couple of cubes and a hexagon at the base
S = []
S = shapes.cuboid([0,0,0], 0.5) + shapes.cuboid([0,0.5+0.25,0], 0.25)
hexagon = [vec(np.cos(th)*1.5, -0.5, np.sin(th)*1.5) for th in np.linspace(0, np.pi*2, 7)]
S.append(hexagon)

# Camera transformation
np.random.seed(10)
view = (trans_3d(vec(0,0,-2.5)) @
        rotx_3d(0.2) @
        roty_3d(np.random.uniform(-1,1)*0.7))
# Projection matrix (switch between perspective and parallel)
#proj = perspective(geom.radians(60), rect_aspect(viewport))
proj = geom.parallel(rect_aspect(viewport)) #1, 1)
# Viewport transformations 3d -> 2d
Sv = view_3d(S, view, proj, viewport, clip=True) # clip True/False enables/disables viewport clipping

plotter = plotters.AxiDrawClient() # Socket connection to axidraw_server.py
#plotter = plotters.AxiPlotter() # Direct connection to AxiDraw using axi module
#plotter = plotters.NoPlotter() # Simply draws output

plot.figure('A5')
plot.stroke_rect(viewport, 'r', linestyle=':')
plot.stroke(Sv, 'k')
plot.show(title='Cubes')
