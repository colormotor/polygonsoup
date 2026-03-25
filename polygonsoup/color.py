#!/usr/bin/env python3
#%%
import numpy as np
from math import fmod
import random
import os
import matplotlib.pyplot as plt
from PIL import Image


def image_to_palette(im, n):
    im = im.quantize(n, method=Image.Quantize.MEDIANCUT, kmeans=n).convert('RGB')
    colors = im.getcolors()
    return [np.array(c[1])/255 for c in colors]


def hex(html_color):
    # Remove '#' if present
    if html_color.startswith("#"):
        html_color = html_color[1:]

    # Extract RGB or RGBA components
    if len(html_color) == 6:
        r = int(html_color[:2], 16) / 255.0
        g = int(html_color[2:4], 16) / 255.0
        b = int(html_color[4:6], 16) / 255.0
        return np.array([r, g, b, 1.0])
    elif len(html_color) == 8:
        r = int(html_color[:2], 16) / 255.0
        g = int(html_color[2:4], 16) / 255.0
        b = int(html_color[4:6], 16) / 255.0
        a = int(html_color[6:8], 16) / 255.0
        return np.array([r, g, b, a])
    else:
        raise ValueError("Invalid HTML color format")


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

    chroma = r - g if g < b else r - b  # r - (g < b ? g : b);
    h = abs(K + (g - b) / (6 * chroma + 1e-20))
    s = chroma / (r + 1e-20)
    v = r

    return np.array([h, s, v, a])[: len(rgba)]


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

    return np.array([r, g, b, a])[: len(hsva)]


def rgba(rgb, a=1.0):
    return np.concatenate([rgb, np.ones(1) * a])


def fmod1(v):
    if v < 0:
        return v + 1
    if v > 1:
        return v - 1
    return v


# https://www.procjam.com/tutorials/en/color/
def adic(n, offset, s=1, v=1, scale=1):
    hues = []
    size = (1 / n) * scale
    for i in range(n):
        h = i * size
        hues.append(np.array(fmod1(h + offset)))
    return [hsv_to_rgb([h, s, v]) for h in hues]


def split_complementary(h1, split=0.1, s=1, v=1, offset=0.5):
    H = [h1, fmod1(fmod1(h1 + offset) + split), fmod1(fmod1(h1 + offset) - split)]
    return [hsv_to_rgb([h, s, v]) for h in H]


def rectangle(h1, split=0.1, s=1, v=1):
    colors = []
    H = [
        fmod1(h1 + split),
        fmod1(h1 - split),
        fmod1(fmod1(h1 + 0.5) + split),
        fmod1(fmod1(h1 - 0.5) - split),
    ]
    return [hsv_to_rgb([h, s, v]) for h in H]


def perturb(rgb, h_offset=0.01, s_offset=0.01, v_offset=0.01):
    h, s, v = rgb_to_hsv(rgb)
    h = fmod1(h + np.random.normal() * h_offset)
    s = np.clip(
        s - abs(np.random.normal() * s_offset), 0, 1
    )  # np.random.uniform(0, s_offset))
    v = np.clip(
        v - abs(np.random.normal() * v_offset), 0, 1
    )  # np.random.uniform(0, v_offset))
    return hsv_to_rgb([h, s, v])


def randomize(palette, n, h_offset=0.05, s_offset=0.2, v_offset=0.2):
    res = []
    for i in range(n):
        rgb = random.choice(palette)
        res.append(perturb(rgb, h_offset, s_offset, v_offset))
    return res


def brightness_contrast(rgb, brightness=0.0, contrast=1.0):
    rgb = np.array(rgb)
    rgb_adj = (rgb - 0.5) * contrast + 0.5 + brightness
    return np.clip(rgb_adj, 0, 1)


def palette_brightness_contrast(colors, brightness=0.0, contrast=1.0):
    return [brightness_contrast(color, brightness, contrast) for color in colors]


def brightness(rgb, w):
    hsv = rgb_to_hsv(rgb)
    hsv[2] = np.clip(hsv[2] * w, 0, 1)
    return hsv_to_rgb(hsv)


def saturation(rgb, w):
    hsv = rgb_to_hsv(rgb)
    hsv[1] = np.clip(hsv[1] * w, 0, 1)
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

        self.im = cv2.imread(path)[:, :, ::-1]

    def color(self, u, v):
        h, w, _ = self.im.shape
        i = np.clip(int(v * h), 0, h - 1)
        j = np.clip(int(u * w), 0, w - 1)
        return self.im[i, j] / 255

    def random(self, v):
        return self.color(np.random.uniform(), v)

    def display(self, v):
        plt.figure(figsize=(4, 4))
        plt.imshow(self.im)
        plt.plot([-2], [v * self.im.shape[0]], "ko", markersize=4)
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

        plut.figure(3, 1)
        for i in range(self.palette.size()):
            rect = geom.make_rect(i, 0, 1, 1)
            plut.fill_rect(rect, self.color(i))
        plut.show()


class ASEPaletteCollection:
    def __init__(self, path, i=0):
        import autograff.utils as utils

        self.paths = utils.files_in_dir(os.path.expanduser(path), "ase")
        self.select(i)

    def print(self):
        for i, path in enumerate(self.paths):
            print("%d: %s" % (i, utils.filename(path)))

    def select(self, i):
        self.i = i % len(self.paths)
        self.palette = ASEPalette(self.paths[i])

    def random(self):
        return self.palette.random()

    def color(self, i):
        return self.palette.color(i)

    def colors(self):
        return self.palette.colors()


""" A bunch of staticmethods
Color conversions from http://www.easyrgb.com/en/math.php#text7
All take as input and return arrays of size [n, 3]"""
# CIE LAB constants Illuminant= D65
# Simulates noon daylight with correlated color temperature of 6504 K.
Lab_ref_white = np.array([0.95047, 1., 1.08883])
# CIE LAB constants Illuminant= D55
# Simulates mid-morning or mid-afternoon daylight with correlated color
# temperature of 5500 K.
D55_ref_white = np.array([0.9568, 1., 0.9214])
# CIE LAB constants Illuminant= D50
# Simulates warm daylight at sunrise or sunset with correlated color
# temperature of 5003 K. Also known as horizon light.
D50_ref_white = np.array([0.964212, 1., .825188])
# CIE standard illuminant A, . Simulates typical, domestic,
# tungsten-filament lighting with correlated color temperature of 2856 K.
A_ref_white = np.array([1.0985, 1.0000, 0.3558])


def rgb_to_XYZ(rgb):
    arr = np.swapaxes(rgb, 0, 1)
    arr = np.where(arr > 0.04045, ((arr + 0.055) / 1.055) ** 2.4,
                    arr / 12.92)
    matrix = np.array([[0.4124, 0.3576, 0.1805],
                        [0.2126, 0.7152, 0.0722],
                        [0.0193, 0.1192, 0.9505]])
    return np.swapaxes(np.dot(matrix, arr), 0, 1)


def XYZ_to_rgb(XYZ):
    arr = np.swapaxes(XYZ, 0, 1)
    matrix_inv = np.array([[ 3.24062548, -1.53720797, -0.4986286 ],
                            [-0.96893071,  1.87575606,  0.04151752],
                            [ 0.05571012, -0.20402105,  1.05699594]])
    arr = np.dot(matrix_inv, arr)
    arr[arr < 0.] = 0.
    arr = np.where(arr > 0.0031308,
                    1.055 * np.power(arr, 1. / 2.4) - 0.055, arr * 12.92)
    arr[arr > 1.] = 1.
    return np.swapaxes(arr, 0, 1)


def XYZ_to_CIELab(XYZ, ref_white=Lab_ref_white):
    arr = XYZ / ref_white
    arr = np.where(arr > 0.008856, arr ** (1. / 3.),
                    (7.787 * arr) + 16. / 116.)
    x, y, z = arr[:, 0], arr[:, 1], arr[:, 2]
    L, a , b = (116. * y) - 16. , 500.0 * (x - y) , 200.0 * (y - z)
    return np.swapaxes(np.vstack([L, a, b]), 0, 1)


def CIELab_to_XYZ(Lab, ref_white=Lab_ref_white):
    L, a, b = Lab[:, 0], Lab[:, 1], Lab[:, 2]
    y = (L + 16.) / 116.
    x = (a / 500.) + y
    z = y - (b / 200.)
    arr = np.vstack([x, y, z])
    arr = np.where(arr > 0.2068966, arr ** 3.,
                    (arr - 16. / 116.) / 7.787)
    return np.swapaxes(arr, 0, 1) * ref_white


def CIELab_to_CIELch(Lab):
    L, a, b = Lab[:, 0], Lab[:, 1], Lab[:, 2]
    h = np.arctan2(b, a)
    h = np.where(h > 0, h / np.pi * 180.,
                    360. + h / np.pi * 180.)
    c = np.sqrt(a**2 + b**2)
    arr = np.vstack([L, c, h])
    return np.swapaxes(arr, 0, 1)


def CIELch_to_CIELab(Lch):
    L, c, h = Lch[:, 0], Lch[:, 1], Lch[:, 2]
    h_rad = h * np.pi / 180.
    a = c * np.cos(h_rad)
    b = c * np.sin(h_rad)
    arr = np.vstack([L, a, b])
    return np.swapaxes(arr, 0, 1)


def _to2d(x):
    """Ensure array is at least 2D (N,3). Return (arr2d, added_axis_flag)."""
    arr = np.asarray(x, dtype=float)
    if arr.ndim == 1:
        return arr[None, :], True
    return arr, False

def _from2d(y, added_axis):
    """If we added an axis on input, remove it on output."""
    y = np.asarray(y)
    return y[0] if added_axis else y

def rgb_to_lab(rgb):
    rgb2d, added = _to2d(rgb)
    out = XYZ_to_CIELab(rgb_to_XYZ(rgb2d))
    return _from2d(out, added)

def rgb_to_lch(rgb):
    rgb2d, added = _to2d(rgb)
    out = CIELab_to_CIELch(XYZ_to_CIELab(rgb_to_XYZ(rgb2d)))
    return _from2d(out, added)

def lab_to_rgb(Lab):
    Lab2d, added = _to2d(Lab)
    out = XYZ_to_rgb(CIELab_to_XYZ(Lab2d))
    return _from2d(out, added)

def lch_to_rgb(Lch):
    Lch2d, added = _to2d(Lch)
    out = XYZ_to_rgb(CIELab_to_XYZ(CIELch_to_CIELab(Lch2d)))
    return _from2d(out, added)

def golden_angle_lch(n, L=68, C=50, H0=None, phi=137.508):
    H0 = float(H0 if H0 is not None else np.random.uniform(0,360))
    return [lch_to_rgb(np.array([L, C, (H0 + k*phi) % 360])) for k in range(n)]

def adic_lch(n, spokes=3, L=70, C=55, H0=None, jitter=0.0):
    """
    a-dic harmony: 'spokes' evenly spaced hues.
    If n > spokes, colors are round-robin assigned to spokes.
    jitter: small random hue jitter in degrees (e.g., 4) to avoid collisions.
    """
    if H0 is None:
        H0 = float(np.random.uniform(0, 360))
    base_hues = [(H0 + i * 360.0 / spokes) % 360 for i in range(spokes)]
    hues = [base_hues[i % spokes] for i in range(n)]
    if jitter > 0:
        hues = [(h + np.random.normal(0, jitter)) % 360 for h in hues]
    return [lch_to_rgb(np.array([L, C, h])) for h in hues]

# https://www.procjam.com/tutorials/en/color/
def adic_lch(n, L=60, C=80, H0=0.0, scale=1, rand=np.zeros(3), llinear=False, lmin=0.0):
    hues = []
    size = (1 / n) * scale
    if llinear:
        lramp = np.linspace(L*lmin, L, n)
    else:
        lramp = np.ones(n)*L
    for i in range(n):
        h = (i * size)*360
        hues.append(np.array(float(h + H0)%360))
    #print(rand)
    return [lch_to_rgb([lramp[i] + np.random.normal()*rand[0],
                        C+np.random.normal()*rand[1],
                        h+np.random.normal()*rand[2]]) for i, h in enumerate(hues)]


def srgb_to_linear(srgb):
    """
    Convert sRGB (nonlinear) in [0,1] to linear RGB.
    Accepts (...,3) array, returns same shape.
    """
    srgb = np.asarray(srgb, dtype=float)
    mask = srgb <= 0.04045
    lin = np.empty_like(srgb)
    lin[mask] = srgb[mask] / 12.92
    lin[~mask] = ((srgb[~mask] + 0.055) / 1.055) ** 2.4
    return lin


def linear_to_srgb(lin):
    """
    Convert linear RGB in [0,1] to sRGB (gamma-encoded).
    Accepts (...,3) array, returns same shape.
    """
    lin = np.asarray(lin, dtype=float)
    mask = lin <= 0.0031308
    srgb = np.empty_like(lin)
    srgb[mask] = 12.92 * lin[mask]
    srgb[~mask] = 1.055 * (lin[~mask] ** (1/2.4)) - 0.055
    return np.clip(srgb, 0.0, 1.0)

if __name__ == "__main__":
    import os

    palettes = ASEPaletteCollection(os.path.expanduser("~/Dropbox/fontdata/palettes"))
    # 2 3 8(?) 14(meh) 17 (maybe) 18 (greenishlight) 30 (gotta darken) 38 (orange)
    palettes.select(47)
    palettes.palette.plot()
