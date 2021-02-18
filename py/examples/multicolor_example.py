#!/usr/bin/env python3
#%%
import numpy as np
import polygonsoup.geom as geom
import polygonsoup.plot as plot
import polygonsoup.plotters as plotters
from polygonsoup.geom import (radians, degrees,
                              trans_3d, rotx_3d, roty_3d, rotz_3d)
import random
import re

# Example showing overlapping of plots with two different colors

# Generate scaled squares with alternating colors
P = geom.rect_corners(geom.make_centered_rect(np.zeros(2), [1,1]), close=True)
S_red = []
S_blue = []
for i, s in enumerate(np.linspace(1, 0.1, 30)):
    Ps = geom.affine_transform(geom.scaling_2d(s), P)
    if i%2:
        S_red.append(Ps)
    else:
        S_blue.append(Ps)

# Show them together without plotting
plot.figure('A5')
plot.stroke(S_red, 'r')
plot.stroke(S_blue, 'b')
plot.show()

# Get bounding box to force both plots to be scaled similarly
# This is necessary because internally, the plotter object scales the paths
# so they fit the drawing area based on their bounding box.
# Different bounding boxes would result in different scales
box = geom.bounding_box(S_red + S_blue)

# Create a plot for each color
plotter = plotters.AxiDrawClient()
plot.figure('A5', plotter=plotter)
plot.stroke(S_red, 'r')
plot.show(box=box)

# wait for manual pen change
input("Change pen, then press Enter to continue...")

plot.figure('A5', plotter=plotter)
plot.stroke(S_red, 'b')
plot.show(box=box)
