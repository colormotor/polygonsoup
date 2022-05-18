#!/usr/bin/env python3
# Just draws a rectangle for test purposed
#%%
from importlib import reload
import polygonsoup.geom as geom
import polygonsoup.plot as plot
import polygonsoup.plotters as plotters
import polygonsoup.geom as geom
reload(plotters)

# plotter = plotters.AxiDrawClient() # Socket connection to axidraw_server.py
# By default the above tries to find the file "client_settings.json" in the same directory as the script

# plotter = plotters.AxiPlotter() # Direct connection to AxiDraw using axi module
plotter = plotters.NoPlotter() # Simply draws output

plot.figure('A5', plotter=plotter)
plot.stroke(np.array([[0, 0],
                      [2, 0],
                      [2, 1],
                      [0, 1]])*3, closed=True)
plot.show()
