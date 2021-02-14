#!/usr/bin/env python3
# Demonstrates boolean operations and clipping with the clipper library (wrapped)
# http://www.angusj.com/delphi/clipperxecute.htm
#%%

from importlib import reload
import numpy as np
import polygonsoup.geom as geom
import polygonsoup.plot as plot
import polygonsoup.hatch as hatch
from polygonsoup.geom import (vec,
                              shapes)

import polygonsoup.clipper as clip

np.random.seed(100)

# Two circles
S = [shapes.circle(vec(0, 0), 10), shapes.circle(vec(5, 0), 10)]
# A random polyline
n = 30
P = np.vstack([np.linspace(-20, 20, n),
               np.random.uniform(-4, 4, n)]).T

# Hatch intersection
Si = clip.intersection(S[0], S[1])
hatches = hatch.hatch(Si, 0.5)
# Clip a polyline with the two circles
P = clip.difference(P, S, False, True) # Flag P as open

# Add polylines
S += hatches

S += [[P]]


plot.figure('A5')
plot.stroke(S, 'k')
plot.show(title='Boolean operations')
