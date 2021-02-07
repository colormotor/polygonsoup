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
from polygonsoup.geom import (vec,
                              make_rect, rect_aspect,
                              shapes,
                              radians, degrees,
                              trans_3d, rotx_3d, roty_3d,
                              affine_transform, perspective, view_3d)
import polygonsoup.clipper as clip
reload(clip)
import vtk

def load_model(path):
    model = vtk.vtkOBJReader()
    model.SetFileName(path)
    model.Update()
    return model.GetOutput()

def contour_lines(poly_data, normal, num=40):
    normals = vtk.vtkPolyDataNormals()
    normals.SetInputData(poly_data)
    normals.Update()

    center = poly_data.GetCenter()
    bounds = poly_data.GetBounds()
    r = geom.distance(vec(bounds[0], bounds[2], bounds[4]), center)

    plane = vtk.vtkPlane()
    plane.SetOrigin(center)
    plane.SetNormal(normal)

    cutter = vtk.vtkCutter()

    cutter.SetCutFunction(plane)
    cutter.SetInputData(normals.GetOutput())
    cutter.GenerateCutScalarsOn()
    cutter.GenerateValues(num, -r, r)
    cutter.Update()

    strips = vtk.vtkStripper()
    strips.SetInputData(cutter.GetOutput())
    strips.Update()

    poly = strips.GetOutput()
    pts = np.array(poly.GetPoints().GetData())
    lines = np.array(poly.GetLines().GetData())
    n = len(lines)
    res = []
    i = 0
    while i < n:
        res.append([])
        nl = lines[i]
        i += 1
        for c in range(nl):
            res[-1].append(pts[lines[i]])
            i += 1

    return res

#model, pos = load_model('teapot.obj'), vec(0, -0.7, -7.5)
model, pos = load_model('stanford-bunny.obj'), vec(0, -0.08, -0.3)
contours = contour_lines(model, [0,1,0], 80)

# Viewport rect
viewport = make_rect(0, 0, 400, 400)

# Camera transformation
view = (trans_3d(pos) @
        rotx_3d(0.6) @
        roty_3d(np.random.uniform(-1,1)*0.7))
# Perspective matrix
proj = perspective(geom.radians(60), rect_aspect(viewport), 0.1)
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

plot.figure(3, 3)
plot.stroke_rect(viewport, 'r', linestyle=':')
plot.stroke(contours_v, 'k')
# Use this to visualize ordering with filled polygons
# for i, ctr in enumerate(contours_v):
#     plot.fill_stroke(ctr, np.ones(3)*0.5, 'k', zorder=i)
plot.show(title='Teapot')
