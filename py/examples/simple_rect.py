#!/usr/bin/env python3
# Just draws a rectangle for test purposed
#%%
from importlib import reload
import polygonsoup.geom as geom
import polygonsoup.plot as plot
import polygonsoup.plotters as plotters
import polygonsoup.geom as geom
import numpy as np
reload(plotters)

plotter = plotters.AxiDrawClient() # Socket connection to axidraw_server.py
# plotter = plotters.AxiPlotter() # Direct connection to AxiDraw using axi module
# plotter = plotters.NoPlotter() # Simply draws output

plot.figure('A3', plotter=plotter)
# plot.stroke(geom.shapes.circle([4, 4], 1), 'k')
# t = np.linspace(0, np.pi*2, 150)
# for r in np.linspace(1, 0.3, 3):
#     circle = np.vstack([4+np.cos(t)*r, 4+np.sin(t)*r]).T
#     plot.stroke(circle, 'k')
#circle = np.vstack([4+np.cos(t)*0.6, 4+np.sin(t)*0.6]).T
#plot.stroke(circle, 'k')
plot.stroke_rect(geom.make_rect(0, 0, 8, 8), 'k')
plot.show()


#%%
f = open('../server/test.gcode')
lines = f.read().split('\n')
S = []
for l in lines:
    tokens = l.split(' ')
    if tokens[1][0] == 'Z':
        continue
    if tokens[0] == 'G0':
        S.append([])
    S[-1].append([float(tokens[1][1:]), float(tokens[2][1:])])

S = [np.array(P) for P in S]
plot.figure((4,4))
plot.stroke(S, 'k')
plot.show()
