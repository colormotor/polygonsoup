'''
  _   _   _   _   _   _   _   _   _   _   _
 / \ / \ / \ / \ / \ / \ / \ / \ / \ / \ / \
( P | O | L | Y | G | O | N | S | O | U | P )
 \_/ \_/ \_/ \_/ \_/ \_/ \_/ \_/ \_/ \_/ \_/

Plotter-friendly graphics utilities
© Daniel Berio (@colormotor) 2021 - ...

plut - visualization utils (matplotlib-based)
'''


import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Path, PathPatch
from matplotlib.colors import ListedColormap
import matplotlib
import numpy as np
import matplotlib as mpl
import polygonsoup.geom as geom

cfg = lambda: None
cfg.dpi = 100
# Make it a bit prettier
cfg.default_style = 'seaborn-darkgrid'
cfg.plotter = None

class NoPlotter:
    '''Default dummy plotter
      Use plotters.AxiDrawClient or plotters.AxiPlotter to plot something
    '''

    def __init__(self):
        pass

    def _set_bounds(self, w, h):
        pass

    def _stroke(self, P):
        pass

    def _plot(self, title='', padding=0, box=None):
        pass

paper_sizes = {
    'A4': (11.7, 8.3),
    'A3': (16.5, 11.7),
    'A5': (8.3, 5.8)
}

cfg.plotter = NoPlotter()

def set_theme(style=cfg.default_style, fontsize=7):
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
    'xtick.major.pad':10.0,
    'ytick.major.pad':10.0,
    'figure.facecolor': (1.,1.,1.,1),
    'savefig.facecolor': (1.,1.,1.,1),
    'legend.fontsize': fontsize*1.2,
    'xtick.labelsize': fontsize*1.1, #*0.9,
    'ytick.labelsize': fontsize*1.1, #*0.9,
    'xtick.color': '666666',
    'ytick.color': '666666',
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


def stroke(S, clr='k', closed=False, **kwargs):
    if type(S)==list and not S:
        # print('Empty shape')
        return
    if geom.is_compound(S):
        for P in S:
            stroke(P, clr=clr, closed=closed, **kwargs)
        return

    # Send out
    P = [p for p in S]
    if closed:
        P = P + [P[0]]

    cfg.plotter._stroke(P)
    P = np.array(P).T

    if len(P.shape) < 2:
        return

    plt.plot(P[0], P[1], color=mpl.colors.to_rgb(clr), **kwargs)

def fill(S, clr, **kwargs):
    if type(S)==list and not S:
        # print('Empty shape')
        return

    if not geom.is_compound(S):
        S = [S]

    if not S:
        # print('Empty shape')
        return

    path = []
    cmds = []
    for P in S:
        if not len(P):
            continue
        path += [p for p in P] + [P[0]]
        cmds += [Path.MOVETO] + [Path.LINETO for p in P[:-1]] + [Path.CLOSEPOLY]
    if path:
        plt.gca().add_patch(PathPatch(Path(path, cmds), color=clr, fill=True, linewidth=0, **kwargs))

def fill_stroke(S, clr, strokeclr, **kwargs):
    if not S:
        # print('Empty shape')
        return
    if not geom.is_compound(S):
        S = [S]

    path = []
    cmds = []
    for P in S:
        # Send out
        cfg.plotter._stroke(P)

        path += [p for p in P] + [P[0]]
        cmds += [Path.MOVETO] + [Path.LINETO for p in P[:-1]] + [Path.CLOSEPOLY]
    plt.gca().add_patch(PathPatch(Path(path, cmds), facecolor=clr, edgecolor=strokeclr, fill=True,  **kwargs))


def stroke_rect(rect, clr='k', plot=True, **kwargs):
    x, y = rect[0]
    w, h = rect[1] - rect[0]

    # Send out
    if plot:
        cfg.plotter._stroke(geom.rect_corners(rect, close=True))

    plt.gca().add_patch(
        patches.Rectangle((x, y), w, h, fill=False, edgecolor=clr, **kwargs))

def fill_rect(rect, clr, **kwargs):
    x, y = rect[0]
    w, h = rect[1] - rect[0]
    plt.gca().add_patch(
        patches.Rectangle((x, y), w, h, fill=True, facecolor=clr, **kwargs))

def fill_circle(pos, radius, clr, **kwargs):
    plt.gca().add_patch(
        patches.Circle(pos, radius, fill=True, facecolor=clr, **kwargs)) #alpha=alpha, zorder=zorder))

def stroke_circle(pos, radius, clr, **kwargs):
    plt.gca().add_patch(
        patches.Circle(pos, radius, fill=False, edgecolor=clr, **kwargs)) #alpha=alpha, zorder=zorder))

def fill_stroke_circle(pos, radius, clr, strokeclr, **kwargs):
    plt.gca().add_patch(
        patches.Circle(pos, radius, fill=True, facecolor=clr, edgecolor=strokeclr, **kwargs)) #alpha=alpha, zorder=zorder))

def stroke_circle(pos, radius, clr, **kwargs):
    plt.gca().add_patch(
        patches.Circle(pos, radius, fill=False, edgecolor=clr, **kwargs)) #alpha=alpha, zorder=zorder))

def draw_markers(P, color, marker='o', **kwargs):
    P = np.array(P)
    if type(color) == str:
        plt.plot(P[:,0], P[:,1], color, linestyle='None', marker=marker, **kwargs)
    else:
        plt.plot(P[:,0], P[:,1], color=color, linestyle='None', marker=marker, **kwargs)

def draw_line(a, b, clr, **kwargs):
    p = np.vstack([a,b])
    plt.plot(p[:,0], p[:,1], color=clr, solid_capstyle='round', dash_capstyle='round', **kwargs)

def det22(mat):
    return mat[0,0] * mat[1,1] - mat[0,1]*mat[1,0]

def draw_arrow(a, b, clr, alpha=1., head_width=0.5, head_length=None, overhang=0.3, zorder=None, **kwargs):
    if head_length is None:
        head_length = head_width

    linewidth = 1.0
    if 'lw' in kwargs:
        linewidth = kwargs['lw']
    if 'linewidth' in kwargs:
        linewidth = kwargs['linewidth']

    # Uglyness, still does not work
    axis = plt.gca()
    trans = axis.transData.inverted()
    scale = np.sqrt(det22(trans.get_matrix()))*axis.figure.dpi*100
    head_width = (linewidth*head_width)*scale
    head_length = (linewidth*head_length)*scale
    a, b  = np.array(a), np.array(b)
    d = b - a

    draw_line(a, b - geom.normalize(d)*head_length*0.5, clr, linewidth=linewidth)
    plt.arrow(a[0], a[1], d[0], d[1], lw=0.5, overhang=overhang,
              head_width=head_width, head_length=head_length, length_includes_head=True,
              fc=clr, ec='none', zorder=zorder)

def arc_points(a, b, theta, subd=100):
    ''' Get points of an arc between a and b,
        with internal angle theta'''

    a = np.array(a)
    b = np.array(b)
    mp = a + (b-a)*0.5

    if abs(theta) < 1e-9:
        theta = 1e-9

    d = a-b #b-a
    l = np.linalg.norm(d)
    r = l / (np.sin(theta/2)*2)

    h = (1-np.cos(theta/2))*r
    h2 = r-h
    p = np.dot([[0,-1],[1, 0]], d)
    pl = np.linalg.norm(p)
    if pl > 1e-10:
        p = p / np.linalg.norm(p)

    cenp = mp-p*h2
    theta_start = np.arctan2(p[1], p[0])
    A = np.linspace(theta_start-theta/2, theta_start+theta/2, subd)
    arc = np.tile(cenp.reshape(-1,1), (1,subd)) + np.vstack([np.cos(A), np.sin(A)]) * r
    return arc.T

def draw_arc_arrow(a, b, angle, clr, alpha=1., head_width=0.25, head_length=None, overhang=0.3, zorder=None, **kwargs):
    if head_length is None:
        head_length = head_width

    linewidth = 1.0
    if 'lw' in kwargs:
        linewidth = kwargs['lw']
    if 'linewidth' in kwargs:
        linewidth = kwargs['linewidth']

    # Uglyness, still does not work
    axis = plt.gca()
    trans = axis.transData.inverted()
    scale = np.sqrt(det22(trans.get_matrix()))*axis.figure.dpi*100
    head_width = (linewidth*head_width)*scale
    head_length = (linewidth*head_length)*scale
    a, b  = np.array(a), np.array(b)
    d = b - a

    arc_pts = arc_points(a, b, geom.radians(angle))
    d = arc_pts[-1] - arc_pts[-2]
    stroke(arc_pts, clr, lw=linewidth)

    #draw_line(a, b - geom.normalize(d)*head_length*0.5, clr, linewidth=linewidth)
    plt.arrow(b[0], b[1], d[0], d[1], lw=0.5, overhang=overhang,
              head_width=head_width, head_length=head_length, length_includes_head=True,
              fc=clr, ec='none', zorder=zorder)


def set_axis_limits(box, pad=0, invert=True, ax=None, y_limits_only=False):
    # UNUSED
    if ax is None:
        ax = plt.gca()

    xlim = [box[0][0]-pad, box[1][0]+pad]
    ylim = [box[0][1]-pad, box[1][1]+pad]

    ax.set_ylim(ylim)
    ax.set_ybound(ylim)
    if not y_limits_only:
        ax.set_xlim(xlim)
        ax.set_xbound(xlim)

    # Hack to get matplotlib to actually respect limits?
    stroke_rect([geom.vec(xlim[0], ylim[0]), geom.vec(xlim[1], ylim[1])], 'r', plot=False, alpha=0)
    # ax.set_clip_on(True)
    if invert:
        ax.invert_yaxis()

def set_axis_limits(P, pad=0, invert=True, ax=None, y_limits_only=False):
    if ax is None:
        ax = plt.gca()

    if type(P) == tuple or (type(P)==list and len(P)==2):
        box = P
        xlim = [box[0][0]-pad, box[1][0]+pad]
        ylim = [box[0][1]-pad, box[1][1]+pad]
    else:
        if type(P) == list:
            P = np.hstack(P)
        xlim = [np.min(P[0,:])-pad, np.max(P[0,:])+pad]
        ylim = [np.min(P[1,:])-pad, np.max(P[1,:])+pad]
    ax.set_ylim(ylim)
    ax.set_ybound(ylim)
    if not y_limits_only:
        ax.set_xlim(xlim)
        ax.set_xbound(xlim)

    # Hack to get matplotlib to actually respect limits?
    stroke_rect([geom.vec(xlim[0],ylim[0]), geom.vec(xlim[1], ylim[1])], 'r', alpha=0, plot=False)
    # ax.set_clip_on(True)
    if invert:
        ax.invert_yaxis()

def show_drawing(drawing, size='A4', title='', padding=0, plotter=NoPlotter()):
    ''' Plots/draws a axi.Drawing object'''
    figure(size, plotter)
    for path in drawing.paths:
        P = [np.array(p) for p in path]
        stroke(P, 'k')
    show(title, padding)

def figure(size="A5", plotter=NoPlotter(), figscale=1):
    if type(size)==str:
        w, h = paper_sizes[size]
    else:
        w, h = size
    fig = plt.figure(dpi=cfg.dpi)
    wfig, hfig = w*figscale, h*figscale

    fig.set_size_inches(wfig, hfig)
    if plotter is None:
        plotter = NoPlotter()
    cfg.plotter = plotter
    plotter._set_bounds(w, h)
    return fig

def show_shape(S, clr='k', figsize=(5,5), axis=False, box=None, **kwargs):
    if type(S) != list:
        S = [S]
    figure(figsize)
    stroke(S, clr, **kwargs)
    show(axis=axis, box=box)

def show_plot(vals, clr='k', figsize=(5,5), **kwargs):
    figure(figsize)
    for y in vals:
        plt.plot(y, **kwargs)
    plt.show()


def show(title='', padding=0, box=None, axis=False, ydown=True, file='', debug_box=False):
    if title:
        plt.title(title)

    setup(ydown, axis, box, debug_box)

    if file:
        plt.savefig(file, transparent=False)

    cfg.plotter._plot(title, padding, box=box)
    plt.show()

def setup(ydown=True, axis=False, box=None, debug_box=False):
    ax = plt.gca()
    ax.axis('scaled')
    if not axis:
        ax.axis('off')
    else:
        ax.axis('on')
    if ydown:
        ax.invert_yaxis()
    if debug_box and box is not None:
        stroke_rect(box, 'r', plot=False)
    if box is not None:
        set_axis_limits(box, invert=ydown, ax=ax, y_limits_only=False)

categorical_palettes = {
    'Tabular':[
(0.12156862745098039, 0.4666666666666667, 0.7058823529411765),
(0.6823529411764706, 0.7803921568627451, 0.9098039215686274),
(1.0, 0.4980392156862745, 0.054901960784313725),
(1.0, 0.7333333333333333, 0.47058823529411764),
(0.17254901960784313, 0.6274509803921569, 0.17254901960784313),
(0.596078431372549, 0.8745098039215686, 0.5411764705882353),
(0.8392156862745098, 0.15294117647058825, 0.1568627450980392),
(1.0, 0.596078431372549, 0.5882352941176471),
(0.5803921568627451, 0.403921568627451, 0.7411764705882353),
(0.7725490196078432, 0.6901960784313725, 0.8352941176470589),
(0.5490196078431373, 0.33725490196078434, 0.29411764705882354),
(0.7686274509803922, 0.611764705882353, 0.5803921568627451),
(0.8901960784313725, 0.4666666666666667, 0.7607843137254902),
(0.9686274509803922, 0.7137254901960784, 0.8235294117647058),
(0.4980392156862745, 0.4980392156862745, 0.4980392156862745),
(0.7803921568627451, 0.7803921568627451, 0.7803921568627451),
(0.7372549019607844, 0.7411764705882353, 0.13333333333333333),
(0.8588235294117647, 0.8588235294117647, 0.5529411764705883),
(0.09019607843137255, 0.7450980392156863, 0.8117647058823529),
(0.6196078431372549, 0.8549019607843137, 0.8980392156862745)
    ],
'Dark2_8':[
(0.10588235294117647, 0.6196078431372549, 0.4666666666666667),
(0.8509803921568627, 0.37254901960784315, 0.00784313725490196),
(0.4588235294117647, 0.4392156862745098, 0.7019607843137254),
(0.9058823529411765, 0.1607843137254902, 0.5411764705882353),
(0.4, 0.6509803921568628, 0.11764705882352941),
(0.9019607843137255, 0.6705882352941176, 0.00784313725490196),
(0.6509803921568628, 0.4627450980392157, 0.11372549019607843),
(0.4, 0.4, 0.4)
],
'Custom':[
[104.0/255, 175.0/255, 252.0/255],
[66.0/255, 47.0/255, 174.0/255],
[71.0/255, 240.0/255, 163.0/255],
[29.0/255, 104.0/255, 110.0/255],
[52.0/255, 218.0/255, 234.0/255],
[45.0/255, 93.0/255, 168.0/255],
[219.0/255, 119.0/255, 230.0/255],
[165.0/255, 46.0/255, 120.0/255],
[171.0/255, 213.0/255, 51.0/255],
[29.0/255, 109.0/255, 31.0/255],
[143.0/255, 199.0/255, 137.0/255],
[226.0/255, 50.0/255, 9.0/255],
[93.0/255, 242.0/255, 62.0/255],
[94.0/255, 64.0/255, 40.0/255],
[247.0/255, 147.0/255, 2.0/255],
[255.0/255, 0.0/255, 135.0/255],
[226.0/255, 150.0/255, 163.0/255],
[216.0/255, 197.0/255, 152.0/255],
[97.0/255, 8.0/255, 232.0/255],
[243.0/255, 212.0/255, 38.0/255]
],
'Paired_12':[ # Okeish
(0.6509803921568628, 0.807843137254902, 0.8901960784313725),
(0.12156862745098039, 0.47058823529411764, 0.7058823529411765),
(0.6980392156862745, 0.8745098039215686, 0.5411764705882353),
(0.2, 0.6274509803921569, 0.17254901960784313),
(0.984313725490196, 0.6039215686274509, 0.6),
(0.8901960784313725, 0.10196078431372549, 0.10980392156862745),
(0.9921568627450981, 0.7490196078431373, 0.43529411764705883),
(1.0, 0.4980392156862745, 0.0),
(0.792156862745098, 0.6980392156862745, 0.8392156862745098),
(0.41568627450980394, 0.23921568627450981, 0.6039215686274509),
(1.0, 1.0, 0.6),
(0.6941176470588235, 0.34901960784313724, 0.1568627450980392)
],
'Tableau_20':[
(0.12156862745098039, 0.4666666666666667, 0.7058823529411765),
(0.6823529411764706, 0.7803921568627451, 0.9098039215686274),
(1.0, 0.4980392156862745, 0.054901960784313725),
(1.0, 0.7333333333333333, 0.47058823529411764),
(0.17254901960784313, 0.6274509803921569, 0.17254901960784313),
(0.596078431372549, 0.8745098039215686, 0.5411764705882353),
(0.8392156862745098, 0.15294117647058825, 0.1568627450980392),
(1.0, 0.596078431372549, 0.5882352941176471),
(0.5803921568627451, 0.403921568627451, 0.7411764705882353),
(0.7725490196078432, 0.6901960784313725, 0.8352941176470589),
(0.5490196078431373, 0.33725490196078434, 0.29411764705882354),
(0.7686274509803922, 0.611764705882353, 0.5803921568627451),
(0.8901960784313725, 0.4666666666666667, 0.7607843137254902),
(0.9686274509803922, 0.7137254901960784, 0.8235294117647058),
(0.4980392156862745, 0.4980392156862745, 0.4980392156862745),
(0.7803921568627451, 0.7803921568627451, 0.7803921568627451),
(0.7372549019607844, 0.7411764705882353, 0.13333333333333333),
(0.8588235294117647, 0.8588235294117647, 0.5529411764705883),
(0.09019607843137255, 0.7450980392156863, 0.8117647058823529),
(0.6196078431372549, 0.8549019607843137, 0.8980392156862745)
],
'Bold_10':[
(0.4980392156862745, 0.23529411764705882, 0.5529411764705883),
(0.06666666666666667, 0.6470588235294118, 0.4745098039215686),
(0.2235294117647059, 0.4117647058823529, 0.6745098039215687),
(0.9490196078431372, 0.7176470588235294, 0.00392156862745098),
(0.9058823529411765, 0.24705882352941178, 0.4549019607843137),
(0.5019607843137255, 0.7294117647058823, 0.35294117647058826),
(0.9019607843137255, 0.5137254901960784, 0.06274509803921569),
(0.0, 0.5254901960784314, 0.5843137254901961),
(0.8117647058823529, 0.10980392156862745, 0.5647058823529412),
(0.9764705882352941, 0.4823529411764706, 0.4470588235294118)
],
    'Prism_10':[
(0.37254901960784315, 0.27450980392156865, 0.5647058823529412),
(0.11372549019607843, 0.4117647058823529, 0.5882352941176471),
(0.2196078431372549, 0.6509803921568628, 0.6470588235294118),
(0.058823529411764705, 0.5215686274509804, 0.32941176470588235),
(0.45098039215686275, 0.6862745098039216, 0.2823529411764706),
(0.9294117647058824, 0.6784313725490196, 0.03137254901960784),
(0.8823529411764706, 0.48627450980392156, 0.0196078431372549),
(0.8, 0.3137254901960784, 0.24313725490196078),
(0.5803921568627451, 0.20392156862745098, 0.43137254901960786),
(0.43529411764705883, 0.25098039215686274, 0.4392156862745098)
    ],
'ColorBlind_10':[
(0.0, 0.4196078431372549, 0.6431372549019608),
(1.0, 0.5019607843137255, 0.054901960784313725),
(0.6705882352941176, 0.6705882352941176, 0.6705882352941176),
(0.34901960784313724, 0.34901960784313724, 0.34901960784313724),
(0.37254901960784315, 0.6196078431372549, 0.8196078431372549),
(0.7843137254901961, 0.3215686274509804, 0.0),
(0.5372549019607843, 0.5372549019607843, 0.5372549019607843),
(0.6352941176470588, 0.7843137254901961, 0.9254901960784314),
(1.0, 0.7372549019607844, 0.4745098039215686),
(0.8117647058823529, 0.8117647058823529, 0.8117647058823529)
],
    'BlueRed_12':[
(0.17254901960784313, 0.4117647058823529, 0.6901960784313725),
(0.7098039215686275, 0.7843137254901961, 0.8862745098039215),
(0.9411764705882353, 0.15294117647058825, 0.12549019607843137),
(1.0, 0.7137254901960784, 0.6901960784313725),
(0.6745098039215687, 0.3803921568627451, 0.23529411764705882),
(0.9137254901960784, 0.7647058823529411, 0.6078431372549019),
(0.4196078431372549, 0.6392156862745098, 0.8392156862745098),
(0.7098039215686275, 0.8745098039215686, 0.9921568627450981),
(0.6745098039215687, 0.5294117647058824, 0.38823529411764707),
(0.8666666666666667, 0.788235294117647, 0.7058823529411765),
(0.7411764705882353, 0.0392156862745098, 0.21176470588235294),
(0.9568627450980393, 0.45098039215686275, 0.47843137254901963)
    ],
    'plut_categorical_12':[
(1.0, 0.42745098039215684, 0.6823529411764706),
(0.8313725490196079, 0.792156862745098, 0.22745098039215686),
(0.0, 0.7450980392156863, 1.0),
(0.9215686274509803, 0.6745098039215687, 0.9803921568627451),
(0.6196078431372549, 0.6196078431372549, 0.6196078431372549),
(0.403921568627451, 0.8823529411764706, 0.7098039215686275),
(0.8392156862745098, 0.15294117647058825, 0.1568627450980392),
(0.6039215686274509, 0.8156862745098039, 1.0),
(0.8862745098039215, 0.5215686274509804, 0.26666666666666666),
(0.12156862745098039, 0.4666666666666667, 0.7058823529411765),
(1.0, 0.4980392156862745, 0.054901960784313725),
(0.36470588235294116, 0.6941176470588235, 0.35294117647058826)
]

}


default_palette_name = 'plut_categorical_12' #Tableau_20' #'Tabular' #'BlueRed_12' #'Tabular' #'BlueRed_12' #'Tabular' #'BlueRed_12'# 'Paired_12' #OK #'ColorBlind_10' #OK # 'Dark2_8'# OKeish 'Bold_10' #OK #
def get_default_colors():
    return categorical_palettes[default_palette_name]
    #return plt.get_cmap('tab20').colors + plt.get_cmap('tab20b').colors + plt.get_cmap('tab20c').colors
    #return plt.rcParams['axes.prop_cycle'].by_key()['color'] + list(plt.get_cmap('tab20').colors) + list(plt.get_cmap('tab20b').colors) + list(plt.get_cmap('tab20c').colors)

def categorical(name=None):
    if name is None:
        name = palette_name
    return categorical_palettes[name]

def categorical_cmap(name=None):
    return ListedColormap(categorical(name))

def default_color(i):
    clrs = get_default_colors() #plt.rcParams['axes.prop_cycle'].by_key()['color']
    return mpl.colors.to_rgb(clrs[i%len(clrs)])

def default_color_alpha(i, alpha):
    rgb = default_color(i) #default_colors[i%len(default_colors)]
    return list(rgb) + [alpha]

def cmap(v, colormap='turbo'): #'PuRd'):
    c = matplotlib.cm.get_cmap(colormap)
    return c(v)

fig = figure
