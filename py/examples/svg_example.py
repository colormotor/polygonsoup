#!/usr/bin/env python3
# Just draws a rectangle for test purposed
#%%
from importlib import reload
import polygonsoup.geom as geom
import polygonsoup.plot as plot
import polygonsoup.plotters as plotters
import polygonsoup.geom as geom

import polygonsoup.svg as svg
import polygonsoup.hatch as hatch

#plotter = plotters.AxiDrawClient() # Socket connection to axidraw_server.py
plotter = plotters.AxiPlotter() # Direct connection to AxiDraw using axi module
#plotter = plotters.NoPlotter() # Simply draws output

plot.figure('A5', plotter=plotter)

S = svg.load_svg('test.svg')
plot.stroke(S)

hatches = hatch.hatch(S, dist=2, angle=30)
plot.stroke(hatches)

plot.show()
