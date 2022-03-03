#!/usr/bin/env python3
# Demonstrates using VTK contour lines together with clipping
# https://vtk.org/
# http://www.angusj.com/delphi/clipperxecute.htm
#%%
import pdb
from importlib import reload
import numpy as np
import polygonsoup.geom as geom
import polygonsoup.plot as plot
import polygonsoup.hatch as hatch
import polygonsoup.plotters as plotters
from polygonsoup.geom import (vec,
                              make_rect, rect_aspect,
                              shapes,
                              radians, degrees,
                              trans_3d, rotx_3d, roty_3d,
                              affine_transform, perspective, view_3d)
import polygonsoup.clipper as clip

import polygonsoup.vtk_utils as vtku
import vtk

model, pos, name = vtku.load_model('teapot.obj'), vec(0, -0.7, -14.5), 'Teapot'
#model, pos, name = vtku.load_model('teapot.obj'), vec(0, -0.08, -0.55), 'Bunny'
points, triangles = vtku.vtk_to_triangle_mesh(model) # <- See this function (in vtk_utils.py) to check how to load a triangle mesh
# Compose triangles into 3d contours that we can transform with the geom utilities
contours = np.array([[points[j] for j in tri] for tri in triangles])

# Viewport rect
viewport = make_rect(0, 0, 400, 400)
np.random.seed(1950)
# Camera transformation
view = (trans_3d(pos) @
        rotx_3d(0.6) @
        roty_3d(np.random.uniform(-1,1)*1.0))
# Perspective matrix
proj = perspective(geom.radians(30), rect_aspect(viewport), 0.1)
# Viewport transformations 3d -> 2d
contours_v = view_3d(contours, view, proj, viewport, clip=True)

#plotter = plotters.AxiDrawClient(port=9001) # Socket connection to axidraw_server.py
#plotter = plotters.AxiPlotter() # Direct connection to AxiDraw using axi module
plotter = plotters.NoPlotter() # Simply draws output

plot.figure('A3', plotter=plotter)
#plot.stroke_rect(viewport, 'r', linestyle=':')
for i, ctr in enumerate(contours_v):
    plot.stroke(ctr, 'k', closed=True)
plot.show(axis=True) #title=name)
