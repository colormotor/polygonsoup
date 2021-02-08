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
reload(geom); reload(plot); reload(hatch)
import polygonsoup.plotters as plotters
reload(plotters)
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

#model, pos, name = vtku.load_model('teapot.obj'), vec(0, -0.7, -14.5), 'Teapot'
model, pos, name = vtku.load_model('stanford-bunny.obj'), vec(0, -0.08, -0.55), 'Bunny'
contours = vtku.contour_lines(model, [0,1,0], 80)

# Viewport rect
viewport = make_rect(0, 0, 400, 400)

# Camera transformation
view = (trans_3d(pos) @
        rotx_3d(0.6) @
        roty_3d(np.random.uniform(-1,1)*0.7))
# Perspective matrix
proj = perspective(geom.radians(30), rect_aspect(viewport), 0.1)
# Viewport transformations 3d -> 2d
contours_v = view_3d(contours, view, proj, viewport, clip=True)

clip_contours = True
if clip_contours:
    # Hacky clipping procedure
    # It assumes contour lines are generated for Y axis, and a viewing angle from the top
    centroids = [np.mean(z, axis=0) for z in contours]
    ord = [c[1] for c in centroids]
    I = np.argsort(ord)
    contours_v = [contours_v[i] for i in I]

    def clip_sorted(contours):
        res = []
        all = []
        for P in contours[::-1]:
            if all:
                Pc = clip.difference(P, all, False, True)
                all = clip.union(all, P)
            else:
                Pc = [P]
                all = [P]
            res += Pc
        return res

    contours_v = clip_sorted(contours_v)

plotter = plotters.AxiDrawClient() # Socket connection to axidraw_server.py
#plotter = plotters.AxiPlotter() # Direct connection to AxiDraw using axi module
#plotter = plotters.NoPlotter() # Simply draws output

plot.figure('A5', plotter=plotter)
plot.stroke_rect(viewport, 'r', linestyle=':')
plot.stroke(contours_v, 'k')
# Use this to visualize ordering with filled polygons
# for i, ctr in enumerate(contours_v):
#     plot.fill_stroke(ctr, np.ones(3)*0.5, 'k', zorder=i)
plot.show(title=name)
