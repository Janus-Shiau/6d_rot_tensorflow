'''
Copyright (c) 2019 [Jia-Yau Shiau]
Code work by Jia-Yau (jiayau.shiau@gmail.com).
--------------------------------------------------
A very simple example code for converting between the rotation6d and rotation matrix.
'''

import math
from functools import reduce

import numpy as np
import tensorflow as tf
from termcolor import colored

from rot_6d import tf_matrix_to_rotation6d, tf_rotation6d_to_matrix


def euler2mat(z=0, y=0, x=0):
    ''' Return matrix for rotations around z, y and x axes

    Uses the z, then y, then x convention above

    Parameters
    ----------
    z : scalar
       Rotation angle in radians around z-axis (performed first)
    y : scalar
       Rotation angle in radians around y-axis
    x : scalar
       Rotation angle in radians around x-axis (performed last)

    Returns
    -------
    M : array shape (3,3)
       Rotation matrix giving same rotation as for given angles

    Code fork from "https://afni.nimh.nih.gov/pub/dist/src/pkundu/meica.libs/nibabel/eulerangles.py".
    '''
    Ms = []
    if z:
        cosz = math.cos(z)
        sinz = math.sin(z)
        Ms.append(np.array(
                [[cosz, -sinz, 0],
                 [sinz, cosz, 0],
                 [0, 0, 1]]))
    if y:
        cosy = math.cos(y)
        siny = math.sin(y)
        Ms.append(np.array(
                [[cosy, 0, siny],
                 [0, 1, 0],
                 [-siny, 0, cosy]]))
    if x:
        cosx = math.cos(x)
        sinx = math.sin(x)
        Ms.append(np.array(
                [[1, 0, 0],
                 [0, cosx, -sinx],
                 [0, sinx, cosx]]))
    if Ms:
        return reduce(np.dot, Ms[::-1])
    return np.eye(3)


if __name__ == "__main__":
    np_mat = []
    for i in range(3):
        np_mat.append(euler2mat(i, i, i))
    
    np_mat = np.array(np_mat)

    tf_mat = tf.convert_to_tensor(np_mat)
    tf_r6d = tf_matrix_to_rotation6d(tf_mat)
    tf_mat_from_r6d = tf_rotation6d_to_matrix(tf_r6d)
    with tf.Session() as sess:
        mat, r6d, mat_from_r6d = sess.run([tf_mat, tf_r6d, tf_mat_from_r6d])
        
        print (colored("[Original Rotation Matrix]", 'yellow'))
        print (mat)
        print (colored("[Rotation 6D from Rotation Matrix]", 'yellow'))
        print (r6d)
        print (colored("[Rotation Matrix from Rotation 6d]", 'yellow'))
        print (mat_from_r6d.reshape(-1,3,3))
