'''
  _   _   _   _   _   _   _   _   _   _   _
 / \ / \ / \ / \ / \ / \ / \ / \ / \ / \ / \
( P | O | L | Y | G | O | N | S | O | U | P )
 \_/ \_/ \_/ \_/ \_/ \_/ \_/ \_/ \_/ \_/ \_/

Plotter-friendly graphics utilities
© Daniel Berio (@colormotor) 2021 - ...

SVG loading
Requires svgpathtools
To install possibly use:
 pip install git+https://github.com/mathandy/svgpathtools.git
Since the pip version does not support rectangles (at this time at least).
'''

# Naive SVG loader
from importlib import reload
import svgpathtools as svg
import numpy as np
import polygonsoup.bezier as bezier
import polygonsoup.geom as geom

def to_pt(c):
    ''' convert complex number to np vector'''
    return np.array([c.real, c.imag])

# Elliptic arc code borrowed from https://github.com/dbrnz/flatmap-ol-maker
def clamp(value, min_value, max_value):
#======================================
    return min(max(value, min_value), max_value)

def svg_angle(u, v):
#===================
    dot = u.real*v.real + u.imag*v.imag
    length = np.sqrt(u.real**2 + u.imag**2)*np.sqrt(v.real**2 + v.imag**2)
    angle = np.arccos(clamp(dot/length, -1, 1))
    if (u.real*v.imag - u.imag*v.real) < 0:
        angle = -angle
    return angle

def elliptic_arc_point(c, r, phi, eta):
#======================================
    return complex(c.real + r.real*np.cos(phi)*np.cos(eta) - r.imag*np.sin(phi)*np.sin(eta),
                   c.imag + r.real*np.sin(phi)*np.cos(eta) + r.imag*np.cos(phi)*np.sin(eta))

def elliptic_arc_derivative(r, phi, eta):
#========================================
    return complex(-r.real*np.cos(phi)*np.sin(eta) - r.imag*np.sin(phi)*np.cos(eta),
                   -r.real*np.sin(phi)*np.sin(eta) + r.imag*np.cos(phi)*np.cos(eta))

def cubic_bezier_control_points(c, r, phi, eta1, eta2):
#======================================================
    alpha = np.sin(eta2 - eta1)*(np.sqrt(4 + 3*np.power(np.tan((eta2 - eta1)/2), 2)) - 1)/3
    P1 = elliptic_arc_point(c, r, phi, eta1)
    d1 = elliptic_arc_derivative(r, phi, eta1)
    Q1 = complex(P1.real + alpha*d1.real, P1.imag + alpha*d1.imag)
    P2 = elliptic_arc_point(c, r, phi, eta2)
    d2 = elliptic_arc_derivative(r, phi, eta2)
    Q2 = complex(P2.real - alpha*d2.real, P2.imag - alpha*d2.imag)
    return (P1, Q1, Q2, P2)

def cubic_beziers_from_arc(arc):#r, phi, flagA, flagS, p1, p2):
#========================================================
    r = arc.radius
    p1 = arc.start
    p2 = arc.end
    phi = geom.radians(arc.rotation)
    flagA = False #arc.large_arc
    flagS = True #arc.sweep
    print(arc)
    r_abs = complex(abs(r.real), abs(r.imag))
    d = complex((p1.real - p2.real), (p1.imag - p2.imag))
    p = complex(np.cos(phi)*d.real/2 + np.sin(phi)*d.imag/2,
              -np.sin(phi)*d.real/2 + np.cos(phi)*d.imag/2)
    p_sq = complex(p.real**2, p.imag**2)
    r_sq = complex(r_abs.real**2, r_abs.imag**2)

    ratio = p_sq.real/r_sq.real + p_sq.imag/r_sq.imag
    if ratio > 1:
        scale = np.sqrt(ratio)
        r_abs = complex(scale*r_abs.real, scale*r_abs.imag)
        r_sq = complex(r_abs.real**2, r_abs.imag**2)

    dq = r_sq.real*p_sq.imag + r_sq.imag*p_sq.real
    pq = (r_sq.real*r_sq.imag - dq)/dq
    q = np.sqrt(max(0, pq))
    if flagA == flagS:
        q = -q

    cp = complex(q * r_abs.real*p.imag/r_abs.imag,
               -q * r_abs.imag*p.real/r_abs.real)
    c = complex(cp.real*np.cos(phi) - cp.imag*np.sin(phi) + (p1.real + p2.real)/2.0,
               cp.real*np.sin(phi) + cp.imag*np.cos(phi) + (p1.imag + p2.imag)/2.0)

    lambda1 = svg_angle(complex(                   1,                     0),
                        complex((p.real - cp.real)/r_abs.real, ( p.imag - cp.imag)/r_abs.imag))
    delta = svg_angle(complex(( p.real - cp.real)/r_abs.real, ( p.imag - cp.imag)/r_abs.imag),
                      complex((-p.real - cp.real)/r_abs.real, (-p.imag - cp.imag)/r_abs.imag))
    delta = delta - 2*np.pi*np.floor(delta/(2*np.pi))
    if not flagS:
        delta -= 2*np.pi
    lambda2 = lambda1 + delta

    t = lambda1
    dt = np.pi/4
    curves = []
    while (t + dt) < lambda2:
        control_points = (cp for cp in cubic_bezier_control_points(c, r_abs, phi, t, t + dt))
        curves.append(svg.CubicBezier(*control_points))
        t += dt
    control_points = (cp for cp in cubic_bezier_control_points(c, r_abs, phi, t, lambda2))
    curves.append(svg.CubicBezier(*(tuple(control_points)[:3]), p2))
    return curves

def to_bezier(piece):
    ''' convert a line or Bezier segment to control points'''
    one3d = 1./3
    if type(piece)==svg.path.Line:
        a, b = to_pt(piece.start), to_pt(piece.end)
        return [[a, a+(b-a)*one3d, b+(a-b)*one3d, b]]
    elif type(piece)==svg.path.CubicBezier:
        return [[to_pt(piece.start),
                to_pt(piece.control1),
                to_pt(piece.control2),
                to_pt(piece.end)]]
    elif type(piece)==svg.path.QuadraticBezier:
        QP0 = to_pt(piece.start)
        QP1 = to_pt(piece.control)
        QP2 = to_pt(piece.end)
        CP1 = QP0 + 2/3 *(QP1-QP0)
        CP2 = QP2 + 2/3 *(QP1-QP2)
        return [[QP0, CP1, CP2, QP2]]
    elif type(piece)==svg.path.Arc:
        bezs = sum([to_bezier(ap) for ap in cubic_beziers_from_arc(piece)], [])
        print('Arc bezs')
        print(bezs)
        return bezs

    raise ValueError

def path_to_bezier(path):
    ''' convert SVG path to a Bezier control points'''
    pieces = sum([to_bezier(piece) for piece in path], [])
    bezier = [pieces[0][0]] + sum([piece[1:] for piece in pieces],[])
    return np.vstack(bezier)

def to_segment(piece):
    ''' convert a line or Bezier segment to control points'''
    return [to_pt(piece.start), to_pt(piece.end)]

def path_to_polyline(path):
    ''' convert SVG path to a Bezier control points'''
    pieces = [to_segment(piece) for piece in path]
    bezier = [pieces[0][0]] + sum([piece[1:] for piece in pieces],[])
    return np.vstack(bezier)


from functools import reduce
def split_compound_paths(paths):
    ''' Split compound paths, since svgpathtools does not do that by default'''
    import re
    split_paths = []
    for path in paths:
        if not path:
            continue
        s = path.d()
        # split at occurrences of moveto commands
        sub_d = filter(None, re.split('[Mm]', s))
        # indices (without moveto) of splits
        lens = [len(list(filter(None, re.split('[A-z]', d))))-1 for d in sub_d]
        # cum sum
        split_inds = [0] + reduce(lambda c, x: c + [c[-1] + x], lens, [0])[1:]
        split_paths += [svg.Path(*path[a:b]) for a, b in zip(split_inds, split_inds[1:])]
    return split_paths

def svg_to_beziers(path):
    ''' Load Bezier curves from a SVG file'''
    paths, attributes = svg.svg2paths(path)
    paths = split_compound_paths(paths)
    beziers = [path_to_bezier(path) for path in paths]
    return beziers

def load_svg(file_path, subd=40):
    paths = svg_to_beziers(file_path)
    S = [bezier.bezier_piecewise(path, subd) for path in paths]
    return S

def load_svg_polylines(path):
    ''' Load Bezier curves from a SVG file'''
    paths, attributes = svg.svg2paths(path)
    paths = split_compound_paths(paths)
    polylines = [path_to_polyline(path) for path in paths]
    return polylines

def load_svg_bezier_chains(file_path, subd=40):
    paths = svg_to_beziers(file_path)
    return [path.T for path in paths]

def save_svg_polylines(S, file_path, closed=False):
    if type(closed)==bool:
        closed = [closed for _ in S]
    S = [geom.close(P) if c else P for P, c in zip(S, closed)]
    paths = [svg.Path(*[svg.Line(start=complex(*a), end=complex(*b)) for a, b in zip(P, P[1:])]) for P in S if len(P) > 1]
    svg.wsvg(paths, filename=file_path)


try:
    #import drawSvg as dsvg
    import drawsvg as dsvg
    color_conv = {'k':'#000000',
                  'w':'#ffffff',
                  'c':'#00ffff'}

    def to_hex(clr):
        if isinstance(clr, str):
            if clr in color_conv:
                return color_conv[clr]
            else:
                if clr[0]=='#':
                    return clr
                return '#000000'
        return '#%02x%02x%02x'%(int(clr[0]*255), int(clr[1]*255), int(clr[2]*255))

    def svg_shape(path, S, closed=False):
        if not isinstance(S, list):
            S = [S]
        for P in S:
            if not len(P):
                continue
            path.M(*(P[0]))
            for p in P[1:]:
                path.L(*(p))
            if closed:
                path.L(*(P[0]))

    class SvgDraw:
        def __init__(self, rect, padding=0, background=None):

            rect = geom.pad_rect(rect, -padding)
            size = geom.rect_size(rect)
            origin = rect[0]
            center = geom.rect_center(rect)

            self.drawing = dsvg.Drawing(*geom.rect_size(rect), origin=rect[0])
            self.g = dsvg.Group(transform='translate(%f,%f) scale(1,-1) translate(%f,%f)'%(*(-center), *center))
            self.drawing.append(self.g)

            if background is not None:
                self.fill_rect(rect, background)

        def stroke(self, S, clr, closed=False, lw=1):
            path = dsvg.Path(stroke=to_hex(clr), stroke_width=lw, fill='none')
            svg_shape(path, S, closed)
            self.g.append(path)

        def fill(self, S, clr):
            path = dsvg.Path(fill=to_hex(clr), stroke='none', fill_rule="evenodd")
            svg_shape(path, S, closed=True)
            self.g.append(path)

        def fill_stroke(self, S, fill_clr, stroke_clr, lw=1):
            path = dsvg.Path(fill=to_hex(fill_clr), stroke=to_hex(stroke_clr), stroke_width=lw, fill_rule="evenodd")
            svg_shape(path, S, closed=True)
            self.g.append(path)

        def fill_rect(self, rect, clr):
            P = np.array(geom.rect_corners(rect))
            self.fill(P, clr)

        def stroke_rect(self, rect, clr, lw=1):
            P = np.array(geom.rect_corners(rect))
            self.stroke(P, clr, closed=True, lw=lw)

        def show(self, w=800):
            self.drawing.setRenderSize(w=w)
            return self.drawing
            #return self.drawing.rasterize()

        def save(self, fname):
            print('saving ' + fname)
            if '.png' in fname:
                self.drawing.save_png(fname)
            else:
                self.drawing.save_svg(fname)

except ModuleNotFoundError:
    print('Could not find drawSvg module')
