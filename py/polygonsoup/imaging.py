'''
  _   _   _   _   _   _   _   _   _   _   _
 / \ / \ / \ / \ / \ / \ / \ / \ / \ / \ / \
( P | O | L | Y | G | O | N | S | O | U | P )
 \_/ \_/ \_/ \_/ \_/ \_/ \_/ \_/ \_/ \_/ \_/

Plotter-friendly graphics utilities
© Daniel Berio (@colormotor) 2021 - ...

imaging - imaging utils
'''


import polygonsoup.geom as geom
import polygonsoup.plut as plut
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

def shape_to_outline(S):
    s = ImageDraw.Outline()
    for P in S:
        s.move(*P[0])
        for p in P[1:]:
            s.line(*p)
        s.close()
    return s

class ShapeRasterizer:
    ''' Helper class to rasterize shapes via PIL'''
    def __init__(self, src_rect, raster_size=512, debug_draw=False):
        if type(raster_size) not in [list, tuple, np.ndarray]:
            raster_size = (raster_size, raster_size)
        elif type(raster_size)==np.ndarray:
            raster_size = raster_size.shape

        dst_rect = geom.make_rect(0, 0, *raster_size) #raster_size, raster_size)
        if src_rect is None:
            src_rect = dst_rect
        self.box = dst_rect #geom.scale_rect(dst_rect, 1)  # 1.2)

        self.mat = geom.rect_in_rect_transform(src_rect, dst_rect)
        self.inv_mat = np.linalg.inv(self.mat)
        self.scale = np.sqrt(np.linalg.det(self.mat))
        self.raster_size = raster_size
        self.debug_draw = debug_draw
        self.context = self.create_context()

    def create_context(self, color=0):
        ''' Create a new image with given size'''
        im = Image.new("L", self.raster_size, color)
        draw = ImageDraw.Draw(im)
        self.context = (im, draw)
        return im, draw

    def set_context(self, context):
        if context is not None:
            self.context = context

    def fill_circle(self, p, r, color=255, context=None):
        self.set_context(context)
        im, draw = self.context
        xy = geom.affine_transform(self.mat, p)
        r = r * self.scale
        draw.ellipse([xy[0] - r, xy[1] - r, xy[0] + r, xy[1] + r], fill=color)

    def fill_circles(self, centers, radii, color=255, context=None):
        self.set_context(context)
        im, draw = self.context
        for p, r in zip(centers, radii):
            self.fill_circle(p, r, color, context)

    def fill_shape(self, S, color=255, context=None):
        self.set_context(context)
        im, draw = self.context
        if type(S) != list:
            S = [S]
        S = geom.affine_transform(self.mat, S)
        draw.shape(shape_to_outline(S), color)

    def stroke_shape(self, S, lw=1, color=255, context=None):
        self.set_context(context)
        im, draw = self.context
        if type(S) != list:
            S = [S]
        S = geom.affine_transform(self.mat, S)
        for P in S:
            for a, b in zip(P, P[1:]):
                draw.line((tuple(a), tuple(b)), fill=255, width=lw, joint='curve')
        #draw.shape(shape_to_outline(S), color)

    def fill_polygon(self, P, color=255, context=None):
        self.set_context(context)
        im, draw = self.context
        P = geom.affine_transform(self.mat, P)
        P = [(float(p[0]), float(p[1])) for p in P]
        draw.polygon(P, fill=color)  # , outline=color)

    def blit(self, context_src, context=None):
        self.set_context(context)
        im, draw = self.context
        draw.bitmap((0,0), context_src[0], 255)

    def contours(self, context=None):
        self.set_context(context)
        im, draw = self.context
        ctrs = find_contours(np.array(im))
        if not ctrs:
            return ctrs
        ctrs = geom.affine_transform(self.inv_mat, ctrs)
        return ctrs

    def get_image(self, context=None):
        self.set_context(context)
        im, draw = self.context
        return np.array(im)
