#!/usr/bin/env python3
# Just draws a rectangle for test purposed
#%%
from importlib import reload
import polygonsoup.geom as geom
import polygonsoup.plot as plot
import polygonsoup.plotters as plotters
import polygonsoup.geom as geom
import polygonsoup.utils as utils

reload(plotters)

#%%
# plotter = plotters.AxiDrawClient() # Socket connection to axidraw_server.py
# By default the above tries to find the file "client_settings.json" in the same directory as the script
stuff = utils.load_pkl('./creativeRoboticsJeremieOCHIN.pickle')
plotter = plotters.NoPlotter() # Simply draws output

plot.figure('A5', plotter=plotter)
for i in range(len(S)):
    plot.stroke(S[i], plot.default_color(i))
plot.show()

#%%
import motor.pd as pd
reload(pd)
import matplotlib.pyplot as plt
data = utils.load_pkl('./datafile.pkl')
data = pd.PD(data, 4.1, 1.5, 0.3, 0.001)
fig = plt.figure(figsize=(7,9))
ax = fig.add_subplot(projection='3d')
ax.plot(data[0,:], data[1,:], data[2,:]) #, c=color_list[cnt])
plt.show()
#%%

#%%
n = 30
dt = 1e-1
kp = 1
kv = 4 * np.sqrt(2. * kp) # Underdamped



#damp_ratio = 4 # Overdamped
#damp_ratio = 1 # Overdamped
pts = np.vstack((xPlot_arr, yPlot_arr, zPlot_arr))
print(pts.shape)
endPlot = spring_path(pts, n, 100, 0.2, dt)



print(endPlot.shape)
ax.plot(endPlot[0,:], endPlot[1,:], endPlot[2,:], c=color_list[cnt])

ax.set_title(fileName)
ax.set_xlabel('X axis')
ax.set_ylabel('Y axis')
ax.set_zlabel('Z axis')



plt.show()
