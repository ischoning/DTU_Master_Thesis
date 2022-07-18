"""
Functions used in the pre-processing of data in event detection.
"""

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import os
import sys
import pickle
from struct import *
import pandas as pd
import seaborn as sns
import mixture
sns.set_style("white")
import warnings
warnings.filterwarnings("ignore")
import collections
from math import pi, cos, sin, cosh, tanh
from scipy.spatial.transform import Rotation as Rot
import cv2
import scipy.interpolate
import scipy.integrate
import scipy.stats
from sklearn.linear_model import RANSACRegressor, TheilSenRegressor, LinearRegression
from sklearn.mixture import GaussianMixture
from scipy.stats import norm
from scipy.signal import argrelextrema
from statistics import median


# Ficks Angle Conversions
def vec2ficks(x, y, z):
    '''
    Calculate ficks angles, corresponding to the given vector, conveniently labelled azimuth and polar angle.
    
    Ficks angles are based on nested gimbals, where the outer gimbal is rotated first about the 
    vertical axis (az) and then about the now rotated horizontal (shoulder) axis (pol). Ficks is normally
    (as here) described using passive rotations, where the second rotation is around an axis rotated 
    by the first rotation. It can, however, be described using active rotations, where the rotation axes are
    fixed in world coordinates; to do so the two rotations are applied in the opposite direction.
    
    Thus, in terms of world coordinates, calculating the resulting gaze vector using the two Ficks
    angles can be calculated as follows:
    
        (x, y, z) = np.dot(rotmat.R('y', az), np.dot(rotmat.R('x', pol), np.array([0, 0, 1])))
        
    or equivalently
    
        (x, y, z) = np.dot(np.matmul(rotmat.R('y', az), rotmat.R('x', pol)), np.array([0, 0, 1])))
        
    where x is the shoulder axis, y the vertical axis and z represents the line of sight.
    
    (Haslwanter, T. (1995). Mathematics of three-dimensional eye rotsations. Vision research, 35(12), 1727-1739.)

    Note that a 3D vector cannot represent a torsion component, so only 2 Ficks angles are returned here.

    This that the calculation here is identical to converting to "adapted 2 spherical coordinates, for 
    which the formula used is based on e.g. https://web.physics.ucsb.edu/~fratus/phys103/Disc/disc_notes_3_pdf.pdf, 
    just adapted to conventional gaze concepts where the polar angle is 0 at the center coordinate
    and azimuth increases in the direction of x, counting from z.

    '''
    r = np.sqrt(x**2 + y**2 + z**2)
    az = np.arctan2(x , z) # Note: If we had done az = 90 - np.arccos(x / r) we would return visual angles
    pol = np.arccos(y / r)

    return np.rad2deg(az), 90 - np.rad2deg(pol)

def visang2vec(h, v, z=1, norm=True):
    '''
    Calculate a directional vector x,y,z from visual angles Ah and Av
    
    The vector returned is the intercept with the direction and a plane at z=z
    
    The formula below has been derived from the vec2visang formula and also 
    deals correctly with negative angles, but breaks down when z~=0(!)
    
    Also, even though h is allowed to rotate fully, v is capped to +/- 90 degrees
    '''

    th2 = np.tan(np.deg2rad(h))**2
    tv2 = np.tan(np.deg2rad(v))**2
    th2tv2 = th2*tv2

    x2 = (th2 + th2tv2) / (1 - th2tv2) * z**2
    y2 = (tv2 + th2tv2) / (1 - th2tv2) * z**2
    
    x = np.sign((h+180)%360-180)*np.sqrt(np.abs(x2))
    y = np.sign((v+180)%360-180)*np.sqrt(np.abs(y2))
    
    if isinstance(h, (collections.Sequence, np.ndarray)):
        z = np.ones(x.shape)
        z[np.where((h+90)%360>180)] = -1
    else:    
        z = z * (-1 if (h+90)%360>180 else 1)
    
    if (norm):
        r = np.sqrt(np.abs(x2) + np.abs(y2) + z**2)
    else:
        r = 1
        
    return x/r, y/r, z/r

# def ficks_rot_mat(theta, phi, psi):
#     """
#     Returns 3x3 rotation matrix given angles theta, phi, psi.
#     """
#     R_Ficks = np.array([[np.cos(theta)*np.cos(phi), np.cos(theta)*np.sin(phi)*np.sin(psi)-np.sin(theta)*np.cos(psi),
#                    np.cos(theta)*np.sin(phi)*np.cos(psi)+np.sin(theta)*np.sin(psi)], 
#                    [np.sin(theta)*np.cos(phi), np.sin(theta)*np.sin(phi)*np.sin(psi)+np.cos(theta)*np.cos(psi),
#                    np.sin(theta)*np.sin(phi)*np.cos(psi)-np.cos(theta)*np.sin(psi)], 
#                    [-np.sin(phi), np.cos(phi)*np.sin(psi), np.cos(phi)*np.cos(psi)]])
#     return R_Ficks