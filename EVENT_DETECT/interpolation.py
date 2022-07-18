"""
Functions used in interpolation in event detection.
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


def interpolate_velocity(t, interp_v, SAMPLE_RATE):
    """
    Returns the peak, avg, and std of intersample velocity of upsampled interpolated data over a given time.
    """  
    # Upsample features
    interp_t = np.linspace(t[0], t[-1], num=np.ceil(500/SAMPLE_RATE).astype(int)*len(t))
    vels = [interp_v(t_i) for t_i in interp_t]
    #vels = [interp_dx(t_i)+interp_dy(t_i) for t_i in interp_t]
    #y_vels = interp_dy(interp_t)
    
    # Calculate features (ignore nans)
    max_vel = np.nanmax(vels)
    avg_vel = np.nanmean(vels)
    std_vel = np.nanstd(vels) 
    
    # Return max and min velocity in y-direction too, for use in Dv calculation
    #max_dy, min_dy = np.nanmax(y_vels), np.nanmin(y_vels)

    return max_vel, avg_vel, std_vel #, max_dy, min_dy


def interpolate(temp, SAMPLE_RATE, feat):
    """
    Interpolate a feature from a dataframe that includes 'time' and feature.
    Returns a dataframe of interpolated data.
    """  
    if feat == '':
        raise Exception("No feature provided. Choose from df features such as x_deg, y_deg, iss, or isv_y")
    if feat == 'x_deg' or 'y_deg':
        k = 3 # cubic
    else:
        k = 2 # quadratic
    
    # Extract feature arrays
    t = np.array(temp['time'])
    x = np.array(temp[feat])

    # Ignore Nans
    nan_i = np.argwhere(np.isnan(x))
    t = np.delete(t, nan_i)
    x = np.delete(x, nan_i)

    # Calculate features
    interp_t = np.linspace(t[0], t[-1], num=np.ceil(500/SAMPLE_RATE).astype(int)*len(t))
    interp = scipy.interpolate.InterpolatedUnivariateSpline(t, x, k=k)
    #interp_x = interp(interp_t)

    return interp
    #return pd.DataFrame({'time':interp_t, feat:interp_x}), interp
