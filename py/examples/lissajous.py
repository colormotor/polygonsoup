"""
Created on Wed Jan 23 00:22:09 2019
Generating simple calligraphic like glyphs with Lissajous curves
@author: Daniel Berio
"""
#%%
import numpy as np
from importlib import reload
import polygonsoup.plot as plot
import polygonsoup.plotters as plotters
reload(plot)

def lissajous(t, a, b, omega, delta):
    return np.vstack([a*np.cos(omega*t + delta),
                      b*np.sin(t)]).T

def lissajous_glyph():
    n = 200
    S = []

    #t = np.linspace(0, np.pi*3.8, n)
    t = np.linspace(0, np.pi*3.8, n)

    delta = np.random.uniform(-np.pi/2, np.pi/2)
    da = np.random.uniform(-np.pi/2, np.pi/2)
    db = np.random.uniform(-np.pi/2, np.pi/2)
    omega = 2.
    for o in np.linspace(0, 0.2, 2):
        a = np.sin(np.linspace(0, np.pi*2, n) + da + o*0.5)*100
        b = np.cos(np.linspace(0, np.pi*2, n) + db + o*1.0)*100
        P = lissajous(t, a,
                         b,
                         omega,
                         delta)
        S.append(P)
    return S, delta, da, db


# Generate
S, delta, da, db = lissajous_glyph()

# Plot
plotter = plotters.AxiDrawClient() # Socket connection to axidraw_server.py
#plotter = plotters.AxiPlotter() # Direct connection to AxiDraw using axi module
# plotter = plotters.NoPlotter() # Simply draws output

plot.figure('A5', plotter=plotter)
plot.stroke(S, 'k')
plot.show(title='pylissajous')






# %%
