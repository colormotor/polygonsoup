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

def main():
    system = axi.LSystem({
        'A': 'A-B--B+A++AA+B-',
        'B': '+A-BB--B-A++A+B',
    })
    d = system.run('A', 3, 60)
    # system = axi.LSystem({
    #     'X': 'F-[[X]+X]+F[+FX]-X',
    #     'F': 'FF',
    # })
    # d = system.run('X', 6, 20)
    d = d.rotate_and_scale_to_fit(12, 8.5, step=90)
    # d = d.sort_paths()
    # d = d.join_paths(0.015)

    # NB: The following needs Cairo to work
    # d.render(bounds=axi.V3_BOUNDS).write_to_png('out.png')

    plotter = plotters.AxiDrawClient('localhost') # Socket connection to axidraw_server.py
    #plotter = plotters.AxiPlotter() # Direct connection to AxiDraw using axi module
    #plotter = plotters.NoPlotter() # Simply draws output
    plot.show_drawing(d, size='A5', title='L-system', plotter=plotter)

if __name__ == '__main__':
    main()


# %%
