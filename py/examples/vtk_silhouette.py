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
reload(geom); reload(plot); reload(hatch)
from polygonsoup.geom import (vec,
                              make_rect, rect_aspect,
                              shapes,
                              radians, degrees,
                              trans_3d, rotx_3d, roty_3d,
                              affine_transform, perspective, view_3d)
import polygonsoup.clipper as clip
reload(clip)
import polygonsoup.vtk_utils as vtku
reload(vtku)

np.set_printoptions(suppress=True)
#model, pos = vtku.load_model('teapot.obj'), vec(0, -1.7, -14.5)
model, pos = vtku.load_model('stanford-bunny.obj'), vec(0, -0.08, -0.55)

# Viewport rect
viewport = make_rect(0, 0, 400, 400)

view = (trans_3d(pos) @
        rotx_3d(0.6) @
        roty_3d(np.random.uniform(-1,1)*0.7))
# Perspective matrix
proj = perspective(geom.radians(30), rect_aspect(viewport), 0.1)

contours = vtku.silhouette(model, view, feature_angle=False, border_edges=False)
contours_v = view_3d(contours, view, proj, viewport, clip=True)

clip_contours = True

plotter = plotters.AxiDrawClient() # Socket connection to axidraw_server.py
#plotter = plotters.AxiPlotter() # Direct connection to AxiDraw using axi module
#plotter = plotters.NoPlotter() # Simply draws output

plot.figure('A5', plotter=plotter)
plot.stroke_rect(viewport, 'r', linestyle=':')
plot.stroke(contours_v, 'k')
# Use this to visualize ordering with filled polygons
# for i, ctr in enumerate(contours_v):
#     plot.fill_stroke(ctr, np.ones(3)*0.5, 'k', zorder=i)
plot.show(title='Silhouette')
