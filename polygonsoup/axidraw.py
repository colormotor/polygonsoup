#!/usr/bin/env python3
import axi
import time
import numpy as np
import matplotlib.pyplot as plt
from polygonsoup import geom, plut
import polygonsoup.simplify as simp
import json
import zmq
import serial, time


def rotate_shape(S, amt=-90):
    return geom.affine_transform(geom.rot_2d(geom.radians(amt)), S)

class Axidraw:
    def __init__(self, src_rect=None, size=(11.7, 8.3), use_mm=False, padding=0.75, pre_transform=np.eye(3)):
        '''
        stc_rect: the input drawing area
        size: size of the paper (in inches)
        padding: padding
        '''
        try:
            self.dev = axi.Device()
            self.dev.max_velocity = 0.5
            self.dev.configure()
        except Exception as e:
            #print(e)
            self.dev = None

        if src_rect is None:
            src_rect = geom.make_rect(0, 0, size[0], size[1])

        self.axi_rect = geom.make_rect(0, 0, size[0], size[1]) # inches
        self.src_rect = src_rect
        self.aximat = geom.rect_in_rect_transform(self.src_rect, self.axi_rect, padding) #, 0.6)
        if use_mm:
            self.aximat = geom.scaling_2d(1.0/25.4) @ self.aximat
        self.pen_offset = np.zeros(2)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.disconnect_axi()
        # Return False to propagate exceptions if any occurred, True to suppress them
        return False


    def preview(self, S):
        if type(S) != list:
            S = [S]
        S = geom.affine_transform(self.aximat, S)
        plut.figure((8, 8))
        plut.stroke_rect(self.axi_rect, 'r')
        for i, P in enumerate(S):
            plut.stroke(P, plut.default_color(i))
        plut.show()

    def disconnect_axi(self):
        self.dev.close()
        self.dev = None

    def axi_pt(self, p):
        return tuple(geom.affine_transform(self.aximat, p) + self.pen_offset)

    def axi_path(self, P, closed):
        Pa = [self.axi_pt(p) for p in P]
        if closed:
            Pa.append(Pa[0])
        Pa = axi.crop_path(Pa, 0. , 0., *tuple(self.axi_rect[1]))[0]
        return Pa

    def home(self):
        if self.dev is None:
            return
        self.dev.home()

    def up(self):
        if self.dev is None:
            return
        self.dev.pen_up()

    def down(self):
        if self.dev is None:
            return
        self.dev.pen_down()

    def enable_motors(self):
        if self.dev is None:
            return
        self.dev.enable_motors()

    def set_pen_up_position(self, pos):
        if self.dev is None:
            print("No axidraw device")
            return
        self.dev.pen_up_position = pos
        self.dev.configure()

    def set_pen_down_position(self, pos):
        if self.dev is None:
            print("No axidraw device")
            return
        self.dev.pen_down_position = pos
        self.dev.configure()

    def disable_motors(self):
        if self.dev is None:
            return
        self.dev.disable_motors()

    def plot(self, S, closed=False,
                        max_velocity=1,
                        penup_speed=200,
                        penup_pos=None,
                        pendown_pos=None,
                        preview=False,
                        pen_offset=np.zeros(2)): #200):
        self.pen_offset = pen_offset

        if preview:
            self.preview(S)
            return

        if self.dev is None:
            print("No axidraw device")
            return
        try:
            if not geom.is_compound(S):
                S = [S]
            self.dev.max_velocity = max_velocity
            self.dev.pen_up_speed = penup_speed
            self.dev.pen_down_speed = penup_speed
            if penup_pos is not None:
                self.dev.pen_up_position = penup_pos
            if pendown_pos is not None:
                self.dev.pen_down_position = pendown_pos
            self.dev.configure()

            for P in S:
                self.dev.pen_up()
                self.dev.goto(*self.axi_pt(P[0]))
                self.dev.pen_down()
                self.dev.run_path(self.axi_path(P, closed))
                self.dev.pen_up()

            self.dev.wait()
        except serial.SerialException:
            print('Axi: Serial error')

    def draw_bounds(self, scale=1):
        Cp = geom.rect_corners(geom.scale_rect(self.src_rect, scale))
        S = []
        S.append(np.array(Cp + [Cp[0]]))
        self.axi_shape(S, max_velocity=3)
        self.home()
        self.close()


