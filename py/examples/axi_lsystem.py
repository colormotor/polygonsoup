''' Lindenmeyer system
adapted from the axi (https://github.com/fogleman/axi) package by Michael Fogleman
This demonstrates how to use the library to generate graphics and plot
the output with the polygonsoup system
'''
#%%
# Code from  the axi (https://github.com/fogleman/axi) example  by Michael Fogleman
import axi
import polygonsoup.plot as plot
import polygonsoup.plotters as plotters
import pdb
from lindemeyer import LSystem

def main():
    system = LSystem({
        'X': 'F+[[X]-X]-F[-FX]+X',
        'F': 'FF',
    })
    S = system.run('X', 3, 25)

    # d = d.sort_paths()
    # d = d.join_paths(0.015)
    plot.figure()

    #plot.stroke(S)
    l = np.inf
    for i in range(len(S)):
        l = min(l, geom.chord_length(S[i]))
        plot.stroke(S[i], plot.default_color(i), lw=2)
    print('Minimum length is %.4f'%l)
    plot.show()
    # NB: The following needs Cairo to work
    # d.render(bounds=axi.V3_BOUNDS).write_to_png('out.png')

    # plotter = plotters.AxiDrawClient() # Socket connection to axidraw_server.py
    #plotter = plotters.AxiPlotter() # Direct connection to AxiDraw using axi module
    #plotter = plotters.NoPlotter() # Simply draws output
    #plot.show_drawing(d, size='A5', title='L-system', plotter=plotter)

if __name__ == '__main__':
    main()


# %%

paths = [[[0,0]]] -> [[[0,0],[x,y]]]

 [[[0,0],[x,y]],
  [[xold, yold]]]

x, y = 0, 0

for task in instructions:
    if task == 'F':
        x = x + length*cos(radians(a))
        y = y + length*sin(radians(a))
        paths[-1].append([x,y])
    ....
    elif task == '[':
        stack.append([[x, y],a])
    elif task == ']':
        position, heading = stack.pop()
        paths.push([position]) # your starting from here with angle a
        x, y = position
        a = heading


pointList.append([x,y])

# elif task == 'B':

# myTurtle.backward(distance)

elif task == '+':

#myTurtle.right(angle)

a += angle

elif task == '-':

#myTurtle.left(angle)

a += -angle

elif task == '[':

#stack.append((myTurtle.position(), myTurtle.heading()))

stack.append([[x, y],a])

#print("save=", x,y,a)

elif task == ']':

#myTurtle.penup()

position, heading = stack.pop()

x = position[0]

y = position[1]

a = heading

pointList.append([x,y])

return pointList
