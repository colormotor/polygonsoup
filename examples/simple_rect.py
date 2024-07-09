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
reload(plot)
plotter = plotters.AxiDrawClient(port=80, raw=True) # Socket connection to axidraw_server.py
# plotter = plotters.AxiPlotter() # Direct connection to AxiDraw using axi module
# plotter = plotters.NoPlotter() # Simply draws output
#%%
plot.figure((300, 300), plotter=plotter, figscale=0.01)
#plot.stroke(geom.shapes.circle([4, 4], 1, subd=200), 'k')
# t = np.linspace(0, np.pi*2, 150)
# for r in np.linspace(1, 0.3, 3):
#     circle = np.vstack([4+np.cos(t)*r, 4+np.sin(t)*r]).T
#     plot.stroke(circle, 'k')
#circle = np.vstack([4+np.cos(t)*0.6, 4+np.sin(t)*0.6]).T
#plot.stroke(circle, 'k')
plot.stroke_rect(geom.make_rect(0, 0, 300, 300), 'k')
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

#%%
plotter.sendln('$110=9000')
plotter.sendln('$111=9000')
#%%
plotter.sendln('$110=3000')
plotter.sendln('$111=3000')
#%%
plotter.sendln('$112=8000')
plotter.sendln('$122=300')

#%%
for i in range(4):
    plotter.pen_down()
    plotter.pen_up()
#
# #plotter.sendln('$24=100')
# plotter.sendln('$$')
#%%
plotter.sendln('G0 Z-32')

#%%
reload(plotters)
plotter = plotters.AxiDrawClient(port=80, raw=True) # Socket connection to axidraw_server.py
#%%

#%%
lines = [np.array([[0, y], [10, y]]) for y in np.linspace(0, 10, 5)]
#%%
plut.figure((500, 500), figscale=0.01, plotter=plotter)
for line in lines[:1]:
    #plut.stroke(line, 'k')
    pump(plotter)

#plot.stroke(geom.shapes.circle([4, 4], 1, subd=200), 'k')
# t = np.linspace(0, np.pi*2, 150)
# for r in np.linspace(1, 0.3, 3):
#     circle = np.vstack([4+np.cos(t)*r, 4+np.sin(t)*r]).T
#     plot.stroke(circle, 'k')
#circle = np.vstack([4+np.cos(t)*0.6, 4+np.sin(t)*0.6]).T
#plot.stroke(circle, 'k')
#plot.stroke_rect(geom.make_rect(0, 0, 600, 600), 'k')
plut.show(box=geom.bounding_box(lines))

#%%
plotter.sendln('$112=8000')
plotter.sendln('$122=200')

#%%
import time
def pump(plotter, reps=4, z=-55):
    for i in range(reps):
        for j in range(13):
            plotter._stroke('G0 Z-53')
            plotter._stroke('G0 Z-53.01')

        plotter._stroke('G0 Z-35')
#pump(plotter, 4)
#%%
plotter.home()
