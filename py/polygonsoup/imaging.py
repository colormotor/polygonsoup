#!/usr/bin/env python3
import autograff.geom as geom
import autograff.plut as plut
import pdb

from PIL import Image, ImageDraw, ImageFont
import cv2
import numpy as np

cfg = lambda: None
cfg.morpho_kernel_size = 5

def find_contours(im, invert=False, thresh=127):
    #im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    if im.dtype != bool:
        if invert:
            ret, thresh = cv2.threshold(im,thresh,256,cv2.THRESH_BINARY_INV)
        else:
            ret, thresh = cv2.threshold(im,thresh,256,0)
    else:
        thresh = cv2.convertScaleAbs(im.astype(float))
    #image, contours, hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    S = []
    for ctr in contours:
        S.append(np.vstack([ctr[:,0,0], ctr[:,0,1]]).T)
    return S

def distance_transform(edgemap, get_inds=True):
    ''' Distance transform of an edge image (e.g. created with Canny)
    Returns distance map, and an array of (flat) indices to closest coordinate.
    Differently from OpenCV this expects edges to be > 0
    '''
    flip = ~edgemap #cv2.threshold(edgemap, 0, 255, cv2.THRESH_BINARY_INV)[1]
    d, lbl = cv2.distanceTransformWithLabels(flip, cv2.DIST_L2, 5, labelType=cv2.DIST_LABEL_PIXEL)

    # cv returns ids of unique labels (not sure why, and if this is actually more efficient on their side)
    # the following is a "recipe" adapted from codes on github, which recovers the indices (flattened)
    # and is efficient enough
    # Faster
    if get_inds:
        idx = np.copy(lbl)
        idx[flip > 0] = 0
        place = np.argwhere(idx > 0)
        near = place[lbl - 1, :]
        idx = near[:, :, 0]*flip.shape[1] + near[:,:,1]
        return d, idx

    # # other methods such as this are sloooow
    # idx = np.zeros(edgemap.shape, dtype=np.intp)
    # idx_func = np.flatnonzero
    # for l in np.unique(lbl):
    #     mask = lbl == l
    #     idx[mask] = idx_func(edgemap * mask)
    # r
    return d

def gradients(bmp):
    dx = cv2.Sobel(bmp, cv2.CV_64F, 1, 0)
    dy = cv2.Sobel(bmp, cv2.CV_64F, 0, 1)
    return dx, dy

def normalized_gradients(bmp):
    dx, dy = gradients(bmp)
    mag = np.sqrt(dx**2.0 + dy**2.0)
    mag[mag==0] = 1
    return dx/mag, dy/mag

def edges(bmp, threshs=[200,255]):
    return cv2.Canny(bmp, *threshs)


# Morphological ops

def kernel(size=None):
    if size is None:
        size = cfg.morpho_kernel_size
    return np.ones((size, size),np.uint8)

def morpho_erode(im, size=None, iterations=1):
    return cv2.erode(im, kernel(size), iterations=iterations)

def morpho_dilate(im, size=None, iterations=1):
    return cv2.dilate(im, kernel(size), iterations=iterations)

def morpho_open(im, size=None, iterations=1):
    return morpho_dilate(morpho_erode(im, size=size, iterations=iterations), size=size, iterations=iterations)

def morpho_close(im, size=None, iterations=1):
    return morpho_erode(morpho_dilate(im, size=size, iterations=iterations), size=size, iterations=iterations)

def morpho_pass(im, size=None, iterations=1):
    return im
