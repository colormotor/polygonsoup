#!/usr/bin/env python3
# Demonstrates plotting inverse kinematics for a URDF/SDF model
# Requires kynpy to be installed https://github.com/neka-nat/kinpy
# Suggested approach:
# conda install -c conda-forge vtk
# pip install git+https://github.com/neka-nat/kinpy.git
#%%

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
                              trans_3d, rotx_3d, roty_3d, rotz_3d, scaling_3d,
                              affine_transform, perspective, view_3d)

from polygonsoup.limb import Limb

np.random.seed(10)
viewport = make_rect(0, 0, 400, 400)

S = []
# Create a circle at the base
noise = 0.0
circle = [vec(np.cos(th)*0.5, np.sin(th)*0.5) +
          np.random.uniform(-noise, noise, 2)
            for th in np.linspace(0, np.pi*2, 80)]
S.append(geom.to_3d_plane(circle, geom.plane_xy, vec(0, 0, 0)))
circle = S[-1]

# Random initial pose
#limb = limb.Limb('simple_arm.sdf', 'arm_wrist_roll')
limb = Limb('./kuka_iwa.urdf', 'lbr_iiwa_link_7')
limb.fk(limb.random_joints())
S.append(limb.frame_positions())

k_orientation = 0.5
soft = False
# reach first point along circle
q = limb.q
x = limb.end_effector()
xh = [*circle[0], *x[3:]]
xhs = [x + (xh - x)*t for t in np.linspace(0, 1, 50)]
for xh in xhs:
    x = limb.end_effector()
    if soft:
        dq = limb.ik_null(q, x, xh, k_orientation=k_orientation)
    else:
        dq = limb.ik(q, x, xh)
    q = q + dq
    limb.fk(q)
    S.append(limb.frame_positions())

# track circle
xh = [*circle[0], *x[3:]]
for p in circle:
    x = limb.end_effector()
    xh[:3] = p
    if soft:
        dq = limb.ik_null(q, x, xh, k_orientation=k_orientation)
    else:
        dq = limb.ik(q, x, xh)
    q = q + dq
    limb.fk(q)
    S.append(limb.frame_positions())


# Camera transformation
view = (geom.zup_basis() # Robot has Z up
        @ trans_3d(vec(-1.2, 0, -0.2))
        @ roty_3d(0.4)
        @ rotz_3d(np.random.uniform(-1,1)*0.7)
        )
# Perspective matrix
proj = perspective(geom.radians(60), rect_aspect(viewport), 0.1)
# Viewport transformations 3d -> 2d
Sv = view_3d(S, view, proj, viewport, clip=False) # clip True/False enables/disables viewport clipping
# Draw
#plotter = plotters.AxiDrawClient() # Socket connection to axidraw_server.py
#plotter = plotters.AxiPlotter() # Direct connection to AxiDraw using axi module
plotter = plotters.NoPlotter() # Simply draws output

plot.figure('A5', plotter=plotter)
#plot.stroke_rect(viewport, 'r', linestyle=':')
plot.stroke(Sv[0], 'r', linewidth=0.5, alpha=0.5)
plot.stroke(Sv[1:], 'k', linewidth=0.5, alpha=0.5)
if soft:
    title = 'IK null-space'
else:
    title = 'IK damped'
plot.show(title=title)
