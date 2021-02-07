#!/usr/bin/env python3
# Demonstrates boolean operations and clipping with the clipper library (wrapped)
# http://www.angusj.com/delphi/clipperxecute.htm
#%%

from importlib import reload
import numpy as np
import polygonsoup.geom as geom
import polygonsoup.plot as plot
import polygonsoup.hatch as hatch
reload(geom); reload(plot); reload(hatch)
from polygonsoup.geom import (vec,
                              shapes)
import polygonsoup.clipper as clip
reload(clip)

# Two circles
S = [shapes.circle(vec(0,0), 10), shapes.circle(vec(5,0), 10)]
# Hatch intersection
Si = clip.intersection(S[0], S[1])
hatches = hatch.hatch(Si, 0.5)
# Clip a polyline with the two circles
poly = np.vstack([np.linspace(-20, 20, 20), np.random.uniform(-4, 4, 20)]).T
segs = clip.difference(poly, S, False, True)

# Add polylines
S += hatches
S += [[segs]]

plot.figure(3, 3)
plot.stroke(S, 'k')
plot.show(title='Boolean operations')
