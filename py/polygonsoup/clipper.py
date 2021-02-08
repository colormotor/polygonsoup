#!/usr/bin/env python3
# Wrapper around the pyclipper package https://github.com/greginvm/pyclipper,
# which wraps this great C++ lib http://www.angusj.com/delphi/clipper.php
import numpy as np
import pyclipper as clip
import pdb
from polygonsoup.geom import is_compound

cfg = lambda: None
cfg.scale = 10000

def ensure_list(v):
    if type(v)==list and not v:
        return v
    if not is_compound(v):
        return [v]
    return v

def polynode_contours(n, contours):
    if n.Contour:
        contours.append(n.Contour)
    for c in n.Childs:
        polynode_contours(c, contours)

def conv_from(S):
    if type(S)==clip.PyPolyNode:
        contours = []
        polynode_contours(S, contours)
        return [[np.array(p)/cfg.scale for p in P] for P in contours]
    else:
        return [[np.array(p)/cfg.scale for p in P] for P in S]

def conv_to(S):
    return tuple([tuple([(int(x*cfg.scale),int(y*cfg.scale)) for x, y in P]) for P in S])
    #return [[(int(x*cfg.scale),int(y*cfg.scale)) for x, y in P.T] for P in S] # tuple([tuple([(int(x*cfg.scale),int(y*cfg.scale)) for x, y in P.T]) for P in S])


def offset(S, amt, join_type='miter', end_type='closed_polygon', miter=2):
    join_types = {'round': clip.JT_ROUND,
                  'square': clip.JT_SQUARE,
                  'miter': clip.JT_MITER}
    # http://www.angusj.com/delphi/clipper/documentation/Docs/Units/ClipperLib/Types/EndType.htm
    end_types = {'closed_polygon': clip.ET_CLOSEDPOLYGON,
                'closed_line': clip.ET_CLOSEDLINE,
                'open_round': clip.ET_OPENROUND,
                'open_square': clip.ET_OPENSQUARE,
                'open_butt': clip.ET_OPENBUTT
                 }
    pco = clip.PyclipperOffset()
    pco.MiterLimit = miter
    #if closed:
    #    closed = clip.ET_CLOSEDPOLYGON
    S = ensure_list(S)

    S = conv_to(S)
    for P in S:
        pco.AddPath(P, join_types[join_type], end_types[end_type])
    res = pco.Execute(amt*cfg.scale)
    return conv_from(res)

def op(op_type, A, B, a_closed=True, b_closed=True, clip_type='nonzero'):
    pc = clip.Pyclipper()
    A = ensure_list(A)
    B = ensure_list(B)
    if type(A) != list:
        A = [A]
    if type(B) != list:
        B = [B]
    cliptypes = {'nonzero': clip.PFT_NONZERO,
                 'evenodd': clip.PFT_EVENODD}
    optypes = {'intersection': clip.CT_INTERSECTION,
               'union': clip.CT_UNION,
               'difference': clip.CT_DIFFERENCE}

    A = conv_to(A)
    B = conv_to(B)

    try:
        pc.AddPaths(A, clip.PT_SUBJECT, a_closed)
        pc.AddPaths(B, clip.PT_CLIP, b_closed)
        if (not a_closed) or (not b_closed):
            solution = pc.Execute2(optypes[op_type], cliptypes[clip_type], cliptypes[clip_type])
        else:
            solution = pc.Execute(optypes[op_type], cliptypes[clip_type], cliptypes[clip_type])
        res = conv_from(solution)
    except clip.ClipperException as e:
        print(e)
        #pdb.set_trace()
        print('Clipper failed, returning first term')
        return conv_from(A)
    return res

def intersection(A, B, a_closed=True, b_closed=True, clip_type='nonzero'):
    A = ensure_list(A)
    B = ensure_list(B)
    return op('intersection', A, B, a_closed, b_closed, clip_type)

def union(A, B, a_closed=True, b_closed=True, clip_type='nonzero'):
    A = ensure_list(A)
    B = ensure_list(B)
    if not A:
        return B
    if not B:
        return A
    return op('union', A, B, a_closed, b_closed, clip_type)

def difference(A, B, a_closed=True, b_closed=True, clip_type='nonzero'):
    A = ensure_list(A)
    B = ensure_list(B)
    if not A:
        return []
    if not B:
        return A
    return op('difference', A, B, a_closed, b_closed, clip_type)

def multi_union(shapes, clip_type='nonzero'):
    A = []
    for B in shapes:
        A = union(A, B, clip_type=clip_type)
    return A
