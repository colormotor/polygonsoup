# TURTLE DEMO
#%%
from importlib import reload
import polygonsoup.plot as plot

import polygonsoup.plotters as plotters


import polygonsoup.geom as geom
import polygonsoup.turtle as turtle
reload(turtle)

import polygonsoup.plotters as plotters
# plotter = plotters.AxiDrawClient() # Socket connection to axidraw_server.py

# plotter = plotters.AxiPlotter() # Direct connection to AxiDraw using axi module

plotter = plotters.NoPlotter() # Simply draws output

# t = turtle.Turtle()
# for i in range(36):
#     t.right(10)
#     t.square(3)

t = turtle.Turtle()
for i in range(36):
    t.right(10)
    t.circle(3, steps=11)

# t.forward(5)
# t.right(90)
# t.forward(5)
# t.up()
# t.left(90)
# t.forward(5)
# t.down()
# t.forward(2)

# plot
plot.figure('A5', plotter=plotter)
plot.stroke(t.paths)
plot.show()
