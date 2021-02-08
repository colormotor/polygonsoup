#!/usr/bin/env python3
# Demonstrates plotting forward kinematics for a URDF/SDF model
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
reload(geom); reload(plot); reload(hatch)
from polygonsoup.geom import (vec,
                              make_rect, rect_aspect,
                              shapes,
                              radians, degrees,
                              trans_3d, rotx_3d, roty_3d, rotz_3d, scaling_3d,
                              affine_transform, perspective, view_3d)

import polygonsoup.limb as limb
reload(limb)


viewport = make_rect(0, 0, 400, 400)

S = []
# Random sine waves on joint angles
limb = limb.Limb('simple_arm.sdf', 'arm_wrist_roll')
freqs = np.random.uniform(0.1, 1., limb.num_frames())
for t in np.linspace(0, np.pi*4, 150):
    limb.fk(np.sin(t*freqs))
    S.append(limb.frame_positions())

# Create a circle at the base
circle = [vec(np.cos(th)*0.5, np.sin(th)*0.5) for th in np.linspace(0, np.pi*2, 37)]
S.append(geom.to_3d_plane(circle, geom.plane_xy, vec(0, 0, 0)))

# Camera transformation
view = (geom.zup_basis() # Robot has Z up
        @ trans_3d(vec(-3.8, 0, -0.4))
        @ roty_3d(0.7)
        @ rotz_3d(np.random.uniform(-1,1)*0.7)
        )
# Perspective matrix
proj = perspective(geom.radians(60), rect_aspect(viewport), 0.1)
# Viewport transformations 3d -> 2d
Sv = view_3d(S, view, proj, viewport, clip=True) # clip True/False enables/disables viewport clipping
# Draw
plot.figure('A5')
plot.stroke_rect(viewport, 'r', linestyle=':')
plot.stroke(Sv, 'k', linewidth=0.5)
plot.show(title='FK')
