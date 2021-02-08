#!/usr/bin/env python3
# Just draws a rectangle for test purposed
#%%
from importlib import reload
import polygonsoup.geom as geom
import polygonsoup.plot as plot
import polygonsoup.plotters as plotters
import polygonsoup.geom as geom
import polygonsoup.svg as svg

#plotter = plotters.AxiDrawClient('localhost') # Socket connection to axidraw_server.py
plotter = plotters.AxiPlotter() # Direct connection to AxiDraw using axi module
#plotter = plotters.NoPlotter() # Simply draws output

S = svg.load_svg('test.svg')
plot.figure('A5', plotter=plotter)
plot.stroke(S)
plot.show(title='SVG')
