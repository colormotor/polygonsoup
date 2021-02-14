#!/usr/bin/env python3
# Matplotlib wrapper.

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Path, PathPatch
import numpy as np
import matplotlib as mpl
import polygonsoup.geom as geom
import polygonsoup.plotters as plotters

cfg = lambda: None
cfg.dpi = 100
# Make it a bit prettier
cfg.default_style = 'seaborn-darkgrid'
cfg.plotter = None

paper_sizes = {
    'A4': (11.7, 8.3),
    'A3': (16.5, 11.7),
    'A5': (8.3, 5.8)
}

def set_theme(style=cfg.default_style, fontsize=6):
    if style:
        # https://towardsdatascience.com/a-new-plot-theme-for-matplotlib-gadfly-2cffc745ff84
        # Place `gadfly.mplstyle` in `~/.matplotlib/stylelib`
        try:
            plt.rcParams.update(plt.rcParamsDefault)
            plt.style.use(style)
            #print('Setting style ' + style)
        except OSError:
            print('Style ' + style + ' not found')
    #plt.style.use('seaborn') #fivethirtyeight') #seaborn-poster') #seaborn-poster')
    #return

    params = {
    # Illustrator compatible fonts
    # http://jonathansoma.com/lede/data-studio/matplotlib/exporting-from-matplotlib-to-open-in-adobe-illustrator/
    'pdf.fonttype': 42,
    'ps.fonttype': 42,
    'axes.titlesize': fontsize+1,
    'font.family': 'Times New Roman',
    'text.color': 'k',
    'lines.markersize': 2,
    'lines.linewidth': 0.75,
    'lines.markeredgewidth':0.25,
    'axes.labelsize': fontsize,
    'axes.facecolor': (1.,1.,1.,1),
    'xtick.major.pad':0.0,
    'ytick.major.pad':0.0,
    'figure.facecolor': (1.,1.,1.,1),
    'savefig.facecolor': (1.,1.,1.,1),
    'legend.fontsize': 4,
    'xtick.labelsize': fontsize*0.9,
    'ytick.labelsize': fontsize*0.9,
    'xtick.color': '333333',
    'ytick.color': '333333',
    'axes.edgecolor' : '666666',
    'axes.grid': True,
    'grid.color': 'cccccc',
    'grid.alpha': 1.,#'dfdfdf',
    'grid.linestyle': ':',
    'grid.linewidth' : 0.25,
    'figure.figsize': [3, 3],
    }


    mpl.rcParams.update(params)

set_theme()


def stroke(S, clr='k', linewidth=0.75, alpha=1., zorder=None):
    if type(S)==list and not S:
        # print('Empty shape')
        return
    if geom.is_compound(S):
        for P in S:
            stroke(P, clr, linewidth, alpha=alpha)
        return

    # Send out
    cfg.plotter._stroke(S)

    P = np.array(S).T
    plt.plot(P[0], P[1], clr, linewidth=linewidth, zorder=zorder)

def fill(S, clr, alpha=1., zorder=None):
    if not S:
        # print('Empty shape')
        return
    if not geom.is_compound(S):
        S = [S]

    path = []
    cmds = []
    for P in S:
        path += [p for p in P] + [P[0]]
        cmds += [Path.MOVETO] + [Path.LINETO for p in P[:-1]] + [Path.CLOSEPOLY]
    plt.gca().add_patch(PathPatch(Path(path, cmds), color=clr, alpha=alpha, fill=True,  linewidth=0, zorder=zorder))

def fill_stroke(S, clr, strokeclr, linewidth=0.75, alpha=1., zorder=None):
    if not S:
        # print('Empty shape')
        return
    if not geom.is_compound(S):
        S = [S]

    path = []
    cmds = []
    for P in S:
        # Send out
        cfg.plotter.stroke(P)

        path += [p for p in P] + [P[0]]
        cmds += [Path.MOVETO] + [Path.LINETO for p in P[:-1]] + [Path.CLOSEPOLY]
    plt.gca().add_patch(PathPatch(Path(path, cmds), facecolor=clr, edgecolor=strokeclr, alpha=alpha, fill=True,  linewidth=linewidth, zorder=zorder))


def stroke_rect(rect, clr='k', alpha=1., linestyle=None, zorder=None, plot=True):
    x, y = rect[0]
    w, h = rect[1] - rect[0]

    # Send out
    if plot:
        cfg.plotter._stroke(geom.rect_corners(rect, close=True))

    plt.gca().add_patch(
        patches.Rectangle((x, y), w, h, fill=False, linestyle=linestyle, edgecolor=clr, zorder=zorder, alpha=alpha))

def fill_rect(rect, clr, alpha=1., zorder=None):
    x, y = rect[0]
    w, h = rect[1] - rect[0]
    plt.gca().add_patch(
        patches.Rectangle((x, y), w, h, fill=True, facecolor=clr, alpha=alpha, zorder=zorder))

def fill_circle(pos, radius, clr, alpha=1., zorder=None):
    plt.gca().add_patch(
        patches.Circle(pos, radius, fill=True, facecolor=clr, alpha=alpha, zorder=zorder))

def set_axis_limits(box, pad=0, invert=True, ax=None, y_limits_only=False):
    # UNUSED
    if ax is None:
        ax = plt.gca()

    xlim = [box[0][0]-pad, box[1][0]+pad]
    ylim = [box[0][1]-pad, box[1][1]+pad]
    ax.set_ylim(ylim)
    ax.set_ybound(ylim)

    # Hack to get matplotlib to actually respect limits?
    stroke_rect([geom.vec(xlim[0], ylim[0]), geom.vec(xlim[1], ylim[1])], 'r', alpha=0, plot=False)
    # ax.set_clip_on(True)
    if invert:
        ax.invert_yaxis()

def show_drawing(drawing, size='A4', title='', padding=0, plotter=plotters.NoPlotter()):
    ''' Plots/draws a axi.Drawing object'''
    figure(size, plotter)
    for path in drawing.paths:
        P = [np.array(p) for p in path]
        stroke(P, 'k')
    show(title, padding)

def figure(size="A5", plotter=plotters.NoPlotter()):
    if type(size)==str:
        w, h = paper_sizes[size]
    else:
        w, h = size
    fig = plt.figure(dpi=cfg.dpi)
    fig.set_size_inches(w, h)
    if plotter is None:
        plotter = plotters.NoPlotter()
    cfg.plotter = plotter
    plotter._set_bounds(w, h)
    return fig

def show(title='', padding=0, axis=False, ydown=True, file=''):
    if title:
        plt.title(title)

    ax = plt.gca()
    ax.axis('scaled')
    if not axis:
        ax.axis('off')
    else:
        ax.axis('on')
    if ydown:
        ax.invert_yaxis()
    if file:
        plt.savefig(file, transparent=True)

    cfg.plotter._plot(title, padding)
    plt.show()
