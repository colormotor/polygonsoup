#!/usr/bin/env python3
# Just draws a rectangle for test purposed
#%%
from importlib import reload
import numpy as np
import polygonsoup.geom as geom
import polygonsoup.plot as plot
import polygonsoup.plotters as plotters
import polygonsoup.geom as geom

import polygonsoup.bezier as bezier

reload(bezier)

#plotter = plotters.AxiDrawClient() # Socket connection to axidraw_server.py
#plotter = plotters.AxiPlotter() # Direct connection to AxiDraw using axi module
plotter = plotters.NoPlotter() # Simply draws output

plot.figure('A5', plotter=plotter)

Cp = np.array([[0, 0], [0.5, 0], [0.5, 0.5], [1, 0.5]])
P = bezier.bezier(Cp, np.linspace(0, 1, 50))
plot.stroke(P, linewidth=2)
bezier.plot_control_polygon(Cp) # This won't go to AxiDraw

# Cp = np.array([[0, 0], [0.5, 0], [0.5, 0.5], [1, 0.5],
#                [1.5, 0.5], # Forces continuity
#                [1.5, 0.], [0.75, 0.2]])
# P = bezier.bezier_piecewise(Cp, subd=100)
# plot.stroke(P, linewidth=2)
# bezier.plot_control_polygon(Cp) # This won't go to AxiDraw

plot.show()
