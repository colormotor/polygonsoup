#!/usr/bin/env python3
''' Wrapper around kinpy: https://github.com/neka-nat/kinpy
with custom IK solutions'''

import numpy as np
import kinpy as kp
import transformations as tf
from polygonsoup.geom import trans_3d
import copy

def transform_to_vec(tsm):
    return np.array([*tsm.pos, *tsm.rot])

def vec_to_transform(x):
    return kp.Transform(x[3:], x[:3])

def quaternion_log( q ):
    """Implements the logarithmic map, which converts a quaternion to axis-angle
        representation. Joao Silverio
     """
    n = np.linalg.norm(q)
    if n > 0.00001:
        q = q/n

    v = q[3]
    u = q[0:3]
    a = np.zeros(3)
    nu = np.linalg.norm(u)

    if nu>0.0:
        a = np.arccos(v)*u/nu

    return a

def project( a, b ):
    return (np.dot(a,b)/np.dot(b,b))*b

def quaternion_diff( q1, q2 ):
    # q1 - q2
    if  np.dot(q1, q2) < 0:
        q2 = q2*-1
    qe = tf.quaternion_multiply( q1, tf.quaternion_inverse(q2) )

    return quaternion_log( qe ) * 2.0

def pose_diff(xh, x):
    xe = np.zeros(6)
    xe[0:3] = xh[0:3] - x[0:3]
    q = x[3:]
    qh = xh[3:]
    xe[3:] = quaternion_diff(qh, q)

    return xe

def damped_pinv(M, l=1.0E-8): #0.2): #1.0E-8):
    M = np.mat(M)
    lambdaI = np.eye(M.shape[0])*l
    # this formulation allows for 0 lambda (Deo & Walker 93)
    return np.array( M.T * np.linalg.inv( M*M.T + lambdaI ) )

def weighted_pinv(X, w, reg=1.0E-8):
    lambdaI = np.eye(X.shape[0])*reg
    W = np.diag(w)
    return W @ X.T @ np.linalg.inv(X @ W @ X.T + lambdaI)

# H is the objective function to be used to optimize in the limb's null space
# partial derivative for H=(1/n) * sum( (q - qhat) / (qmax - qmin) )^2
def deltaH(q, qmin, qmax, qhat):
  return (1.0 / q.size) * 2.0 * (q - qhat) / (qmax - qmin) * (qmax - qmin)

# gradient of H
def gradH(q, qh, joint_min, joint_max):
    dH = np.zeros(q.size)
    for i in range(q.size):
        dH[i] = deltaH(q[i], joint_min[i], joint_max[i], qh[i])
    return dH

import pdb

class Limb:
    def __init__(self, path_or_data, end_link_name):

        if type(path_or_data) == str:
            path = path_or_data
            if 'urdf' in path:
                self.chain = kp.build_serial_chain_from_urdf(open(path).read(), end_link_name)
                #self.chain = kp.build_chain_from_urdf(open(path).read())
            elif 'sdf' in path:
                chain = kp.build_chain_from_sdf(open(path).read())
                #pdb.set_trace()
                self.chain =  kp.chain.SerialChain(chain, end_link_name + "_frame") #, chain._root.name)

        else: # assume urdf
            data = path_or_data
            self.chain = kp.build_serial_chain_from_urdf(data, end_link_name)
        
        print(self.chain)

        # "natural" pose
        self.fk(np.zeros(self.num_frames()))

        self.joint_min = -np.pi * np.ones(self.num_frames())
        self.joint_max = np.pi * np.ones(self.num_frames())
        self.joint_mean = (self.joint_min + self.joint_max)/2

    def num_frames(self):
        return len(self.chain.get_joint_parameter_names())

    def frame_positions(self):
        P = []
        for x in self.pose:
            mat = trans_3d(x[:3]) @ tf.quaternion_matrix(x[3:])
            P.append(mat[:3,-1])
        return P

    def get_fk_pose(self, q):
        tsm = self.chain.forward_kinematics(q, end_only=False)
        return [transform_to_vec(trans) for k, trans in tsm.items()]

    def fk(self, q):
        self.q = q
        self.pose = self.get_fk_pose(q)

    def end_effector(self):
        return np.array(self.pose[-1])

    def jacobian(self, q):
        J = kp.jacobian.calc_jacobian(self.chain, q)
        return J

    def position_jacobian(self, q):
        J = self.jacobian(q)
        return J[0:3,:]

    def orientation_jacobian(self, q):
        J = self.jacobian(q)
        return J[3:,:]

    def clamp_joints(self, q, eps=1e-7):
        return np.maximum( np.minimum(q, self.joint_max - eps), self.joint_min + eps)

    def ik(self, q, x, xh, kp=1, kpa=0.0, reg=0.001):
        ''' Damped pseudoinverse solution'''
        dx = pose_diff(xh, x)
        xpe  = dx[:3] * kp
        xae = dx[3:] * kpa
        xe = np.concatenate([xpe, xae])
        J  = self.jacobian(q)
        dq = damped_pinv(J, reg) @ xe
        return dq

    def ik_null(self, q, x, xh,
              w=None, # optional joint weights
              k_orientation=0.1, # orientation weight in null space
              kp=1, kpa=0,
              reg=0.0):
        ''' Weighted solution using gradient projection to track orientation
        Similar to Berio, Calinon, Leymarie (2017) Learning dynamic graffiti strokes with a compliant robot
        https://www.doc.gold.ac.uk/autograff/post/papers/Berio-IROS2016.pdf
        '''
        if w is None:
            w = np.ones_like(q)
        dx = pose_diff(xh, x)

        xe  = dx[:3] * kp
        xae = dx[3:] * kpa

        J  = self.position_jacobian(q)
        Jinv  = weighted_pinv(J, w, reg)
        dq = Jinv @ xe

        # gradient projection for
        Ja = self.orientation_jacobian(q)
        Jainv = weighted_pinv(Ja, w, reg)
        N = np.eye(q.size) - J.T @ Jinv.T
        dqa = N @ Jainv @ xae

        dq += dqa

        return dq

    def random_joints(self, safe_range=0.9):
        return self.joint_mean + np.random.uniform(0, 1, self.joint_mean.size)*(self.joint_max - self.joint_min)*0.5*safe_range

    def ik_guess( self, xh, qh=None, k=0.1, eps = 0.001, maxiter=1000, w=None ):
        '''TODO: Test me'''
        q = np.array(self.q)

        for i in range(maxiter):
            for j in range(maxiter):
                x = self.get_fk_pose(q)
                dq = self.ik(q, x, xh, qh=qh, k=k, w=w)
                q = q + dq
                x2 = self.get_fk_pose(q)
                err = np.linalg.norm( pose_diff(x2, xh))
                #print err
                if err < eps:
                    print("Found solution after " + str(i) + " iteratons:")
                    print("Error: " + str(err))
                    return q

            print("Err: " + str(err))
            q = self.random_joints(0.7)
        print("Could not find solution")
        return None
