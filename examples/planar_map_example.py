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
reload(geom)


# Two circles
S = [shapes.circle(vec(0, 0), 10), shapes.circle(vec(5, 0), 10)]
# A random polyline
n = 30
P = np.vstack([np.linspace(-20, 20, n),
               np.random.uniform(-4, 4, n)]).T
S += [P]

planar_map = geom.compute_planar_map(S)
faces = geom.planar_map_faces(planar_map)

plot.figure()
plot.stroke(S)
f = geom.face_vertices(geom.find_face(planar_map, [7,0]))
for face in faces: #[f]: #faces[0]]:
    plot.stroke(face, 'r')
    plot.stroke(hatch.hatch(face, np.random.uniform(0.1, 0.3), np.random.uniform(0, 45)))
plot.show()
