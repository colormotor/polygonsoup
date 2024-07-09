from __future__ import division

import time
import math
import numpy as np
import os, sys
import matplotlib.pyplot as plt
import networkx as nx

from polygonsoup.contrib.ttfquery.findsystem import linuxFontDirectories, findFonts
from polygonsoup.contrib.ttfquery import describe
from polygonsoup.contrib.ttfquery import glyphquery
import polygonsoup.contrib.ttfquery.glyph as glyph
import polygonsoup.utils as utils
import polygonsoup.geom as geom
import matplotlib.pyplot as plt

def system_fonts():
    return findFonts()
    
def font_name(font_path):
    font = describe.openFont(font_path)
    return describe.shortName(font)[0]

def glyph_width(font_path, char):
    font = describe.openFont(font_path)
    return glyphquery.width(font, glyphquery.glyphName(font, char))

def glyph_height(font_path):
    font = describe.openFont(font_path)
    return glyphquery.charHeight(font)

# TTF outline decomposition adapted from ttfquery
def decomposeOutline( contour, steps=3, ds=2 ):
    """Decompose a single TrueType contour to a line-loop

    In essence, this is the "interpretation" of the font
    as geometric primitives.  I only support line and
    quadratic (conic) segments, which should support most
    TrueType fonts as far as I know.

    The process consists of first scanning for any multi-
    off-curve control-point runs.  For each pair of such
    control points, we insert a new on-curve control point.

    Once we have the "expanded" control point array we
    scan through looking for each segment which includes
    an off-curve control point.  These should only be
    bracketed by on-curve control points.  For each
    found segment, we call our integrateQuadratic method
    to produce a set of points interpolating between the
    end points as affected by the middle control point.

    All other control points merely generate a single
    line-segment between the endpoints.
    """
    # contours must be closed, but they can start with
    # (and even end with) items not on the contour...
    # so if we do, we need to create a new point to serve
    # as the midway...
    if len(contour)<3:
        return ()
    set = contour[:]
    def on( record ):
        """Is this record on the contour?
        
        record = ((Ax,Ay),Af)
        """
        return record[-1] == 1

    def merge( first, second):
        """Merge two off-point records into an on-point record"""
        ((Ax,Ay),Af) = first 
        ((Bx,By),Bf) = second
        return (((Ax+Bx)/2.0),((Ay+By))/2.0),1
    # create an expanded set so that all adjacent
    # off-curve items have an on-curve item added
    # in between them
    last = contour[-1]
    expanded = []
    for item in set:
        if (not on(item)) and (not on(last)):
            expanded.append( merge(last, item))
        expanded.append( item )
        last = item
    result = []
    last = expanded[-1]
    while expanded:
        assert on(expanded[0]), "Expanded outline doesn't have proper format! Should always have either [on, off, on] or [on, on] as the first items in the outline"
        if len(expanded)>1:
            if on(expanded[1]):
                # line segment from 0 to 1
                #points = [expanded[1]]
                if result:
                    #print(expanded)
                    points = [result[-1], expanded[0][0]]
                    P = geom.uniform_sample(np.array(points), ds)
                    Pp = P*[1,-1]
                    #plt.plot(Pp[:,0], Pp[:,1], 'b', linewidth=0.3)
                    points = [(px, py) for px, py in P[1:]] #[1:]]
                    result.extend( points )
                else:
                    #
                    result.append( expanded[0][0] )
                del expanded[:1]
            else:
                if len(expanded) == 2:                          #KH
                    assert on(expanded[0]), """Expanded outline finishes off-curve""" #KH
                    points = [result[-1], expanded[1][0]]
                    P = geom.uniform_sample(np.array(points), ds)
                    points = [(px, py) for px, py in P] #[1:]]
                    result.extend( points )

                    #result.append( expanded[1][0] )         #KH
                    del expanded[:1] 
                    break
                
                if result:
                    points = [result[-1], expanded[0][0]]
                    P = geom.uniform_sample(np.array(points), ds)
                    Pp = P*[1,-1]
                    #plt.plot(Pp[:,0], Pp[:,1], 'm', linewidth=0.3)
                    points = [(px, py) for px, py in P[1:-1]] #[1:]]
                    result.extend( points )

                
                assert on(expanded[2]), "Expanded outline doesn't have proper format!"
                points = integrateQuadratic( expanded[:3], steps = steps )
                P = geom.uniform_sample(np.array(points), ds)
                points = [(px, py) for px, py in P[:-1]] #[1:]]
                result.extend( points )
                del expanded[:2]
        else:
            assert on(expanded[0]), """Expanded outline finishes off-curve"""

            points = [result[-1], expanded[0][0]]
            P = geom.uniform_sample(np.array(points), ds)
            Pp = P*[1,-1]
            #plt.plot(Pp[:,0], Pp[:,1], 'm', linewidth=0.3)
            points = [(px, py) for px, py in P[1:]] #[1:]]
            result.extend( points )
            
            # points = [result[-1], expanded[0][0]]
            # P = geom.uniform_sample(np.array(points).T, ds).T
            # points = [(px, py) for px, py in P] #[1:]]
            # result.extend( points )
            
            #result.append( expanded[0][0] )
            del expanded[:1]
    result.append( result[-1] )
    return result

def integrateQuadratic( points, steps=3 ):
    """Get points on curve for quadratic w/ end points A and C

    Basis Equations are taken from here:
        http://www.truetype.demon.co.uk/ttoutln.htm

    This is a very crude approach to the integration,
    everything is coded directly in Python, with no
    attempts to speed up the process.

    XXX Should eventually provide adaptive steps so that
        the angle between the elements can determine how
        many steps are used.
    """
    step = 1.0/steps
    ((Ax,Ay),_),((Bx,By),_),((Cx,Cy),_) = points
    result = [(Ax,Ay)]
    ### XXX This is dangerous, in certain cases floating point error
    ## can wind up creating a new point at 1.0-(some tiny number) if step
    ## is sliaghtly less than precisely 1.0/steps
    for t in np.arange( step, 1.0, step ):
        invT = 1.0-t
        px = (invT*invT * Ax) + (2*t*invT*Bx) + (t*t*Cx)
        py = (invT*invT * Ay) + (2*t*invT*By) + (t*t*Cy)
        result.append( (px,py) )
    # the end-point will be added by the next segment...
    #result.append( (Cx,Cy) )
    return result


def glyph_shape_uniform(font_path, char, steps=40, ds=2):
    font = describe.openFont(font_path)
    
    g = glyph.Glyph(glyphquery.glyphName(font, char))
    
    contours = g.calculateContours(font)
    S = []
    for contour in contours:
        P = np.array(decomposeOutline(contour, steps=steps, ds=ds))*[1,-1]
        S.append(P)
    return S
    

def glyph_shape(font_path, char, steps=40):
    font = describe.openFont(font_path)
    
    g = glyph.Glyph(glyphquery.glyphName(font, char))
    
    contours = g.calculateContours(font)
    S = []
    for contour in contours:
        P = np.array(glyph.decomposeOutline(contour, steps=steps))*[1,-1]
        S.append(P)
    return S
    
class FontDatabase:
    def __init__(self, path):
        path = os.path.expanduser(path)
        self.font_paths = utils.files_in_dir(path, ['ttf']) #, 'otf'])
        self.db = {str(font_name(self.font_paths[i]), 'utf-8'): i for i in range(self.num_fonts)}

    @property
    def num_fonts(self):
        return len(self.font_paths)

    def __getitem__(self, i):
        return self.get_path(i)

    def get_font_height(self, i):
        return glyph_height(self.get_path(i))
    
    def char_width(self, i, char):
        return glyph_width(self.get_path(i), char)

    def get_font_name(self, i):
        return str(font_name(self.get_path(i)))

    def find_font_name(self, name):
        name = name.lower()
        for key in self.db.keys():
            if name in key.lower():
                return key
        print('Could not find %s in font database'%name)
        return ''

    def get_path(self, i):
        if type(i) == str:
            return self.font_paths[self.db[i]]

        i = max(0, min(self.num_fonts-1, i))
        return str(self.font_paths[i])

    def get_shape(self, font, char, steps=40):
        if type(font)==str:
            font = self.db[font]
        return glyph_shape(self.font_paths[font], char, steps)

    def get_shape_sampled(self, font, char, target_height=150, steps=40):
        if type(font)==str:
            font = self.db[font]
        ds = self.get_font_height(font)/target_height
        return glyph_shape_uniform(self.font_paths[font], char, steps, ds)

    def list_font_names(self):
        for i in range(self.num_fonts):
            print('%d: %s'%(i,self.get_font_name(i)))
