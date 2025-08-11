#!/usr/bin/env python3
from scipy.spatial.transform import Rotation as R
import numpy as np
from . import geom

def affine_euler(xyz, degrees=False):
    return R.from_euler('xyz',
                xyz,
                 degrees=degrees).as_matrix()

    # return (geom.rotx_3d(rot[0]) @
    #         geom.roty_3d(rot[1]) @
    #         geom.rotz_3d(rot[2]))

class ArcBall:
    '''
    Arcball rotation

    ## Basic usage

    ```
    # Initialization
    a = ArcBall()

    # every frame
    a.update(w, h)
    a.drag(mouse_pressed, mouse_position)

    # apply transformation
    apply_matrix(translation @ a.affine)

    ```

    ## Example usage with a mouse event callback
    ```
    # every frame
    a.update(w, h)

    # on mouse drag
    a.drag(True, mouse_position)

    # on mouse release
    a.unclick()
    ```
    '''
    def __init__(self):
        self.p_click = np.zeros(3)        # Saved click vector
        self.epsilon = 1.0e-5
        self.rot = R.from_quat([0, 0, 0, 1])
        self.dragging = False

    def from_euler(self, rot, degrees=True):
        self.rot = R.from_euler('xyz',
                                rot, degrees=degrees)

    def click(self, p):
        self.p_click = self.map_to_sphere(p)

    def unclick(self):
        self.dragging = False

    def update(self, w, h):
        self.w_scale = 1.0 / ((w - 1.0) * 0.5)
        self.h_scale = 1.0 / ((w - 1.0) * 0.5)

    def constrain(self, v, axis):
        axis = np.array(axis)
        v = np.array(v)
        cv = axis * np.dot(v, axis)
        cv = v - cv
        return cv / np.linalg.norm(cv)

    def drag(self, dragging, p, eps=1e-5):
        if not dragging:
            self.dragging = False
            return
        if not self.dragging:
            self.click(p)
        self.dragging = True
        p_drag = self.map_to_sphere(p) # Drag vector
        d_drag = p_drag - self.p_click

        c = np.cross(self.p_click, p_drag) # Perpendicular axis of rotation
        if np.linalg.norm(c) < eps:
            return

        q_drag = np.concatenate([c, [np.dot(self.p_click, p_drag)]])

        r_drag = R.from_quat(q_drag)
        self.rot = r_drag * self.rot
        self.p_click = p_drag

    def map_to_sphere(self, p):
        v3 = np.array([p[0] * self.w_scale - 1.0,
                       1.0 - (p[1] * self.h_scale),
                       0])
        mag = np.dot(v3, v3)
        if mag > 1.0:
            v3 /= np.sqrt(mag)
        else:
            v3[2] = np.sqrt(1.0 - mag)

        return v3

    @property
    def affine(self):
        ''' Get affine transform with rotation'''
        rot = np.eye(4)
        rot[:3, :3] = self.rot.as_matrix()
        return rot

    def euler_xyz(self, degrees=True):
        ''' Get rotation as euler angles'''
        return self.rot.as_euler('xyz', degrees=degrees)
