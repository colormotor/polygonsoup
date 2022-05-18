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

model, pos, name = vtku.load_model('teapot.obj'), vec(0, -0.7, -14.5), 'Teapot'
# model, pos, name = vtku.load_model('stanford-bunny.obj'), vec(0, -0.08, -0.55), 'Bunny'
contours = vtku.contour_lines(model, [0,1,0], 60)

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
contours_v = [geom.smoothing_spline(0, np.array(P), ds=0.5, degree=2) for P in contours_v if len(P) > 1]
# def join_bruteforce(S):
#     G = nx.Graph()
#     visited = set()
#     for i, P in enumerate(S):
#         visited.add(i)
#         res.append(P)
#         dists = []
#         for j, Q in enumerate(S):
#             if j in visited:
#                 continue
#             p0, p1 = Q[0], Q[-1]

        
clip_contours = True
if clip_contours:
    # Hacky clipping procedure
    # It assumes contour lines are generated for Y axis, and a viewing angle from the top
    centroids = [np.mean(z, axis=0) for z in contours]
    order = [c[1] for c in centroids]

    #order = [np.mean(np.array(z)[:,2]) for z in contours][::-1]
    I = np.argsort(order)
    contours_v = [contours_v[i] for i in I]

    def clip_sorted(contours):
        res = []
        all = []
        for P in contours[::-1]:
            Pc = clip.difference(P, all, False, True)
            all = clip.union(all, P)
            res += Pc
        return res

    contours_v = clip_sorted(contours_v)


#def join_bruteforce(ctrs):

#contours_v = [geom.smoothing_spline(0, np.array(P), ds=2, smooth_k=0) for P in contours_v]
plotter = plotters.AxiDrawClient(port=80, raw=True) #9001) # Socket connection to axidraw_server.py
#plotter = plotters.AxiPlotter() # Direct connection to AxiDraw using axi module
#plotter = plotters.NoPlotter() # Simply draws output

#plot.figure('A3', plotter=plotter)
plot.figure((600, 600), plotter=plotter, figscale=0.01)

#plot.stroke_rect(viewport, 'r', linestyle=':')
#plot.stroke(contours_v, 'k')
for i, ctr in enumerate(contours_v):
    plot.stroke(ctr, plot.default_color(i))
# Use this to visualize ordering with filled polygons
# for i, ctr in enumerate(contours_v):
#     plot.fill_stroke(ctr, np.ones(3)*0.5, 'k', zorder=i)
plot.show(axis=True) #title=name)
