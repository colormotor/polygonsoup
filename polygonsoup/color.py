#!/usr/bin/env python3
#%%
import numpy as np
from math import fmod
import random
import os
from . import plut
import matplotlib.pyplot as plt


def rgb_to_hsv(rgba):
    r, g, b = rgba[:3]
    a = 1
    if len(rgba) > 3:
        a = rgba[-1]

    K = 0
    if g < b:
        g, b = b, g
        K = -1

    if r < g:
        r, g = g, r
        K = -2 / 6 - K

    chroma = r - g if g < b else r - b #r - (g < b ? g : b);
    h = abs(K + (g - b) / (6 * chroma + 1e-20))
    s = chroma / (r + 1e-20)
    v = r

    return np.array([h, s, v, a])[:len(rgba)]


def hsv_to_rgb(hsva):
    h, s, v = hsva[:3]
    a = 1
    if len(hsva) > 3:
        a = hsva[3]

    if s == 0.0:
        r = g = b = v
    else:
        h = fmod(h, 1) / (60.0 / 360.0)
        i = int(h)
        f = h - i
        p = v * (1.0 - s)
        q = v * (1.0 - s * f)
        t = v * (1.0 - s * (1.0 - f))

        if i == 0:
            r, g, b = v, t, p
        elif i == 1:
            r, g, b = q, v, p
        elif i == 2:
            r, g, b = p, v, t
        elif i == 3:
            r, g, b = p, q, v
        elif i == 4:
            r, g, b = t, p, v
        else:
            r, g, b = v, p, q

    return np.array([r,g,b,a])[:len(hsva)]

def rgba(rgb, a=1.0):
    return np.concatenate([rgb, np.ones(1)*a])

def fmod1(v):
    if v < 0:
        return v+1
    if v > 1:
        return v-1
    return v

# https://www.procjam.com/tutorials/en/color/
def adic(n, offset, s=1, v=1):
    hues = []
    size = 1/n
    for i in range(n):
        h = i*size
        hues.append(np.array(fmod1(h+offset)))
    return [hsv_to_rgb([h, s, v]) for h in hues]

def split_complementary(h1, split=0.1, s=1, v=1, offset=0.5):
    colors = []
    H = [h1,
         fmod1(fmod1(h1 + offset) + split),
         fmod1(fmod1(h1 + offset) - split)]
    return [hsv_to_rgb([h, s, v]) for h in H]

def rectangle(h1, split=0.1, s=1, v=1):
    colors = []
    H = [fmod1(h1 + split),
         fmod1(h1 - split),
         fmod1(fmod1(h1 + 0.5) + split),
         fmod1(fmod1(h1 - 0.5) - split)]
    return [hsv_to_rgb([h, s, v]) for h in H]

def perturb(rgb, h_offset=0.05, s_offset=0.05, v_offset=0.05):
    h, s, v = rgb_to_hsv(rgb)
    h = fmod1(h + np.random.normal()*h_offset)
    s = np.clip(s - abs(np.random.normal()*s_offset), 0, 1) # np.random.uniform(0, s_offset))
    v = np.clip(v - abs(np.random.normal()*v_offset), 0, 1) #np.random.uniform(0, v_offset))
    return hsv_to_rgb([h, s, v])

def randomize(palette, n, h_offset=0.05, s_offset=0.2, v_offset=0.2):
    res = []
    for i in range(n):
        rgb = random.choice(palette)
        res.append(perturb(rgb, h_offset, s_offset, v_offset))
    return res

def brightness(rgb, w):
    hsv = rgb_to_hsv(rgb)
    hsv[2] = np.clip(hsv[2]*w, 0, 1)
    return hsv_to_rgb(hsv)

def saturation(rgb, w):
    hsv = rgb_to_hsv(rgb)
    hsv[1] = np.clip(hsv[1]*w, 0, 1)
    return hsv_to_rgb(hsv)

def set_brightness(rgb, w):
    hsv = rgb_to_hsv(rgb)
    hsv[2] = w
    return hsv_to_rgb(hsv)

def set_saturation(rgb, w):
    hsv = rgb_to_hsv(rgb)
    hsv[1] = w
    return hsv_to_rgb(hsv)

class ImagePalette:
    def __init__(self, path):
        import cv2
        self.im = cv2.imread(path)[:,:,::-1]

    def color(self, u, v):
        h, w, _ = self.im.shape
        i = np.clip(int(v*h), 0, h-1)
        j = np.clip(int(u*w), 0, w-1)
        return self.im[i,j]/255

    def random(self, v):
        return self.color(np.random.uniform(), v)

    def display(self, v):
        plt.figure(figsize=(4,4))
        plt.imshow(self.im)
        plt.plot([-2], [v*self.im.shape[0]], 'ko', markersize=4)
        plt.show()

class ASEPalette:
    def __init__(self, path):
        import cautograff as ag
        self.palette = ag.ASEPalette(os.path.expanduser(path))
        self.path = path

    def colors(self):
        return [self.color(i) for i in range(self.palette.size())]

    def random(self):
        return self.palette.random()[:3]

    def color(self, i):
        return self.palette.color(i)[:3]

    def plot(self):
        import autograff.plut as plut
        import autograff.geom as geom
        plut.figure(3,1)
        for i in range(self.palette.size()):
            rect = geom.make_rect(i, 0, 1, 1)
            plut.fill_rect(rect, self.color(i))
        plut.show()


class ASEPaletteCollection:
    def __init__(self, path, i=0):
        import autograff.utils as utils
        self.paths = utils.files_in_dir(os.path.expanduser(path), 'ase')
        self.select(i)

    def print(self):
        for i, path in enumerate(self.paths):
            print('%d: %s'%(i,utils.filename(path)))

    def select(self, i):
        self.i = i%len(self.paths)
        self.palette = ASEPalette(self.paths[i])

    def random(self):
        return self.palette.random()

    def color(self, i):
        return self.palette.color(i)

    def colors(self):
        return self.palette.colors()


if __name__=='__main__':
    import os
    palettes = ASEPaletteCollection(os.path.expanduser('~/Dropbox/fontdata/palettes'))
    # 2 3 8(?) 14(meh) 17 (maybe) 18 (greenishlight) 30 (gotta darken) 38 (orange)
    palettes.select(47)
    palettes.palette.plot()
