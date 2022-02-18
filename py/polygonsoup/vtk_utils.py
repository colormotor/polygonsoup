'''
  _   _   _   _   _   _   _   _   _   _   _
 / \ / \ / \ / \ / \ / \ / \ / \ / \ / \ / \
( P | O | L | Y | G | O | N | S | O | U | P )
 \_/ \_/ \_/ \_/ \_/ \_/ \_/ \_/ \_/ \_/ \_/

Plotter-friendly graphics utilities
© Daniel Berio (@colormotor) 2021 - ...

vtk_utils - Utilities to interface with the Visualization Toolkit (VTK)
'''

import numpy as np
import polygonsoup.geom as geom
import vtk
vec = geom.vec

def load_model(path):
    '''Load a model'''
    model = vtk.vtkOBJReader()
    model.SetFileName(path)
    model.Update()
    return model.GetOutput()


def contour_lines(poly_data, normal, num=40, get_normals=False):
    ''' Generates contour lines for a polydata object'''
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

    polylines = vtk_to_polylines(cutter)
    if get_normals:
        return polylines, vtk_to_normals(cutter)
    else:
        return polylines

def silhouette(poly_data, view, feature_angle=True, border_edges=True):
    ''' Generates silhouette contours for a polydata object'''
    # view matrix to Vtk camera
    campos = -view.T[:3,:3]@view[:3,3]
    camz = -view.T[:3,2]
    camera = vtk.vtkCamera()
    camera.SetPosition(campos)
    camera.SetFocalPoint(campos + camz)
    # debug
    # viewd = geom.vtk_to_np_matrix(camera.GetViewTransformMatrix())
    # print(view-viewd)

    silhouette = vtk.vtkPolyDataSilhouette()
    silhouette.SetInputData(poly_data)
    silhouette.SetCamera(camera)
    silhouette.SetEnableFeatureAngle(feature_angle)
    silhouette.SetBorderEdges(border_edges)
    silhouette.Update()
    contours = vtk_to_polylines(silhouette)
    return contours

def vtk_to_normals(obj):
    '''Convert vtk output to polylines'''
    strips = vtk.vtkStripper()
    strips.SetInputData(obj.GetOutput())
    strips.Update()

    poly = strips.GetOutput()
    normals = np.array(poly.GetNormals().GetData())
    lines = np.array(poly.GetLines().GetData())
    n = len(lines)
    res = []
    i = 0
    while i < n:
        res.append([])
        nl = lines[i]
        i += 1
        for c in range(nl):
            res[-1].append(normals[lines[i]])
            i += 1

    return res

def vtk_to_polylines(obj):
    '''Convert vtk output to polylines'''
    strips = vtk.vtkStripper()
    strips.SetInputData(obj.GetOutput())
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

def vtk_to_ndarray(vmatrix):
    """Return vtkMatrix4x4 or vtkMatrix3x3 elements as numpy array.
    see https://discourse.slicer.org/t/vtk-transform-matrix-as-python-list-tuple-array/11797/2
    """
    from vtk import vtkMatrix4x4
    from vtk import vtkMatrix3x3
    if isinstance(vmatrix, vtkMatrix4x4):
        sz = 4
    elif isinstance(vmatrix, vtkMatrix3x3):
        sz = 3
    else:
        raise RuntimeError("Input must be vtk.vtkMatrix3x3 or vtk.vtkMatrix4x4")
    narray = np.eye(sz)
    vmatrix.DeepCopy(narray.ravel(), vmatrix)
    return narray
