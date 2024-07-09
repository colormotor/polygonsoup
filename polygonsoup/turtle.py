'''
  _   _   _   _   _   _   _   _   _   _   _
 / \ / \ / \ / \ / \ / \ / \ / \ / \ / \ / \
( P | O | L | Y | G | O | N | S | O | U | P )
 \_/ \_/ \_/ \_/ \_/ \_/ \_/ \_/ \_/ \_/ \_/

Plotter-friendly graphics utilities
© Daniel Berio (@colormotor) 2021 - ...

turtle - 2d turtle implementation (adapted from Fogelman's axi module)
'''

import numpy as np
import polygonsoup.geom as geom
vec = geom.vec

class Turtle(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.x = 0
        self.y = 0
        self.h = 0
        self.pen = True
        self._path = [vec(0,0)]
        self._paths = []

    def clear(self):
        self._path = [vec(self.x, self.y)]
        self._paths = []

    @property
    def paths(self):
        paths = list(self._paths)
        if len(self._path) > 1:
            paths.append(self._path)
        return [np.array(P) for P in paths]

    def pd(self):
        self.pen = True
        if len(self._path) < 2:
            self._path = [vec(self.x, self.y)]

    pendown = down = pd

    def pu(self):
        self.pen = False
        if len(self._path) > 1:
            self._paths.append(self._path)
            self._path = []
    penup = up = pu

    def isdown(self):
        return self.pen

    def goto(self, x, y=None):
        if y is None:
            x, y = x
        if self.pen:
            self._path.append(vec(x, y))
        self.x = x
        self.y = y
    setpos = setposition = goto

    def setx(self, x):
        self.goto(x, self.y)

    def sety(self, y):
        self.goto(self.x, y)

    def seth(self, heading):
        self.h = heading
    setheading = seth

    def home(self):
        self.goto(0, 0)
        self.seth(0)

    def fd(self, distance):
        x = self.x + distance * np.cos(geom.radians(self.h))
        y = self.y + distance * np.sin(geom.radians(self.h))
        self.goto(x, y)
    forward = fd

    def bk(self, distance):
        x = self.x - distance * np.cos(geom.radians(self.h))
        y = self.y - distance * np.sin(geom.radians(self.h))
        self.goto(x, y)
    backward = back = bk

    def rt(self, angle):
        self.seth(self.h + angle)
    right = rt

    def lt(self, angle):
        self.seth(self.h - angle)
    left = lt

    def square(self, side):
        for i in range(4):
            self.fd(side)
            self.rt(90)

    def circle(self, radius, extent=None, steps=None):
        if extent is None:
            extent = 360
        if steps is None:
            steps = int(round(abs(2 * np.pi * radius * extent / 180)))
            steps = max(steps, 4)
        cx = self.x + radius * np.cos(geom.radians(self.h + 90))
        cy = self.y + radius * np.sin(geom.radians(self.h + 90))
        a1 = geom.degrees(np.arctan2(self.y - cy, self.x - cx))
        a2 = a1 + extent if radius >= 0 else a1 - extent
        for i in range(steps):
            p = i / float(steps - 1)
            a = a1 + (a2 - a1) * p
            x = cx + abs(radius) * np.cos(geom.radians(a))
            y = cy + abs(radius) * np.sin(geom.radians(a))
            self.goto(x, y)
        if radius >= 0:
            self.seth(self.h + extent)
        else:
            self.seth(self.h - extent)

    def pos(self):
        return vec(self.x, self.y)
    position = pos

    def towards(self, x, y=None):
        if y is None:
            x, y = x
        return geom.degrees(np.arctan2(y - self.y, x - self.x))

    def xcor(self):
        return self.x

    def ycor(self):
        return self.y

    def heading(self):
        return self.h

    def distance(self, x, y=None):
        if y is None:
            x, y = x
        return np.hypot(x - self.x, y - self.y)
