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

# plotter = plotters.AxiDrawClient(port=9001) # Socket connection to axidraw_server.py
plotter = plotters.AxiPlotter() # Direct connection to AxiDraw using axi module
# plotter = plotters.NoPlotter() # Simply draws output

plot.figure('A4', plotter=plotter)
plot.stroke_rect(geom.make_rect(0, 0, *plot.paper_sizes['A4']), 'k')
plot.show(padding=1)

