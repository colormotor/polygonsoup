# TURTLE DEMO
#%%
from importlib import reload
import polygonsoup.plot as plot
import polygonsoup.plotters as plotters
import polygonsoup.geom as geom
import polygonsoup.turtle as turtle

# Socket connection to axidraw_server.py
plotter = plotters.AxiDrawClient()
# Direct connection to AxiDraw using axi module
plotter = plotters.AxiPlotter()
# Simply draws output
plotter = plotters.NoPlotter()

# Logo example:
# to flower
# repeat 36 [right 10 square]
# end
t = turtle.Turtle()
for i in range(36):
    t.right(10)
    t.circle(3, steps=11)

plot.figure('A5', plotter=plotter)
plot.stroke(t.paths)
plot.show()
