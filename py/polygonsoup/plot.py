#!/usr/bin/env python3
# Matplotlib wrapper.

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import matplotlib as mpl

cfg = lambda: None
cfg.dpi = 150
# Make it a bit prettier
cfg.default_style = 'seaborn-darkgrid'

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
    'ps.usedistiller': 'xpdf',
    'ps.fonttype': 42,
    'mathtext.fontset': 'cm',
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


def stroke(S, clr, linewidth=0.75, alpha=1.):
    if not S:
        # print('Empty shape')
        return
    if type(S[0])==list:
        for P in S:
            stroke(P, clr, linewidth, alpha=alpha)
        return
    P = np.array(S).T
    plt.plot(P[0], P[1], clr, linewidth=linewidth)

def stroke_rect(rect, clr, alpha=1., linestyle=None, zorder=None):
    x, y = rect[0]
    w, h = rect[1] - rect[0]
    plt.gca().add_patch(
        patches.Rectangle((x, y), w, h, fill=False, linestyle=linestyle, edgecolor=clr, zorder=zorder, alpha=alpha))

def fill_rect(rect, clr, alpha=1., zorder=None):
    x, y = rect[0]
    w, h = rect[1] - rect[0]
    plt.gca().add_patch(
        patches.Rectangle((x, y), w, h, fill=True, facecolor=clr, alpha=alpha, zorder=zorder))

def figure(w=None, h=None):
    if w==None and h==None:
        fig = plt.figure()
    if h==None:
        h = w
    fig = figure_inches(w, h)
    return fig

def figure_inches(w, h):
    fig = plt.figure(dpi=cfg.dpi)
    fig.set_size_inches(w, h)
    return fig

def setup(axis=False, ydown=True, axis_limits=None, equal=True, ax=None):
    if ax is None:
        ax = plt.gca()

    # save it for later
    if equal:
        #ax.axis('equal')
        ax.axis('scaled')
    if not axis:
        ax.axis('off')
    else:
        ax.axis('on')
    if ydown:
        ax.invert_yaxis()

    if axis_limits is not None:
        set_axis_limits(axis_limits, invert=ydown, ax=ax)


def show(axis=False, ydown=True, file='', axis_limits=None, title=''):
    if title:
        plt.title(title)
    setup(axis, ydown, axis_limits=axis_limits)
    if file:
        plt.savefig(file, transparent=True)
    plt.show()
