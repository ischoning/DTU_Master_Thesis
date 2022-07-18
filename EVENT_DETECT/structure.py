"""
Functions used to build and maintain the structure of dataframes used in event detection.
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
import plotting
from plotting import *
import interpolation
from interpolation import *
import ficks
from ficks import *
import scipy.interpolate
import scipy.integrate
import scipy.stats
from sklearn.linear_model import RANSACRegressor, TheilSenRegressor, LinearRegression
from sklearn.mixture import GaussianMixture
from scipy.stats import norm
from scipy.signal import argrelextrema
from statistics import median


def make_events_df(df, interp_v, SAMPLE_RATE, MIN_FIX_DUR):
    """
    Creates a dataframe of events with features given a dataframe with 'x_deg', 'y_deg', 'event', and 'time'.
    """
    eventStream = []
    stream, stream_i = [], []

    for idx, row in df.iterrows():

        # ignore row if NAN values exist
        #if any(row['time':'y_deg'].values != row['time':'y_deg'].values):
        #    continue

        if len(stream) == 0 or idx == 0:
            prev_row = row
            prev_idx = idx
            stream.append(prev_row)
            stream_i.append(prev_idx)
            continue
            
        # keep increasing the stream as long as the event is the same
        if row.event == prev_row.event:
            prev_row = row
            prev_idx = idx
            stream.append(prev_row)
            stream_i.append(prev_idx)
        # if the event type changes, calculate the features
        else:
            # ignore NAN values when calculating features (though x and y shouldn't have nans because we've interpolated)
            x = [r.x_deg for r in stream if not np.isnan(r.x_deg)]
            y = [r.y_deg for r in stream if not np.isnan(r.y_deg)]
            
            # if the event is a fixation and of duration less than the minimum fixation duration,
            # then combine it with the next event
            duration = stream[-1].time - stream[0].time
            if prev_row.event == 'fix' and duration < MIN_FIX_DUR:
                # relabel from fix to 'other'
                df.loc[stream_i, 'event'] = row.event
                stream.append(prev_row)
                stream_i.append(prev_idx)
                prev_row = row
                prev_idx = idx
                
            # otherwise compute the rest of the features and log the event to stream
            else:
                t = [r.time for r in stream]
                
                # if the event is non-fixation then add neighboring samples to x, y and t
                if prev_row.event != 'fix' and stream_i[0] != 0 and stream_i[-1]+1 < len(df):
                    prev_end = df.loc[stream_i[0]-1]
                    next_start = df.loc[stream_i[-1]+1]
                    x.append(next_start.x_deg)
                    y.append(next_start.y_deg)
                    t.append(next_start.time)
#                     stream.append(next_start)
                    x.insert(0, prev_end.x_deg)
                    y.insert(0, prev_end.y_deg)
                    t.insert(0, prev_end.time)
#                     stream.insert(0, prev_end)
                   
                # if there are no non-nan values then the event will have nans and be labelled as noise
                # TODO: this shouldn't be the case. Interpolated velocity still tell us whether there's movement
                if len(x) == 0 or len(y) == 0:
                    event = 'noise'
                    x.append(np.nan)
                    y.append(np.nan)
                else:
                    event = prev_row.event
                
#                 try:
                duration = t[-1] - t[0]
                amplitude = np.sqrt((x[-1]-x[0])**2 + (y[-1]-y[0])**2)
                dispersion = np.nanstd(x) + np.nanstd(y)
                cov_x, cov_y, cov_xy = np.cov(x), np.cov(y), np.cov(x,y)[0][1]
                std_x, std_y = np.std(x), np.std(y)
                med_x = median(x)
                med_y = median(y)
#                 except: raise ValueError(f'prev event {event}, current event {prev_row.event}, t {t}, x {x}, y {y}')
                
                # if the event is non-fixation and less than 1 deg in amplitude, then combine it with the next event
                # relabel from fix to 'other'
                if prev_row.event != 'fix' and amplitude < 1.5:
                    df.loc[stream_i, 'event'] = row.event
                    stream.append(prev_row)
                    stream_i.append(prev_idx)
                    prev_row = row
                    prev_idx = idx
                    continue
                    
                # calculate velocity:
                # interpolated intersample velocity upsampled to 500 Hz
                max_vel, avg_vel, std_vel = interpolate_velocity(t, interp_v, SAMPLE_RATE)
                
                # interpolated intersample velocity
                max_Dn = np.nanmax([r.Dn for r in stream])
                avg_Dn = np.nanmean([r.Dn for r in stream])
                
                # intersample velocity
                max_iss = np.nanmax([r.iss for r in stream])
                avg_iss = np.nanmean([r.iss for r in stream])
                
                # calculus error
                integral = interp_v.integral(t[0],t[-1])
                calculus_error = np.abs((amplitude-integral)/integral)
                
                # if the calculus error is nan, then that means it couldn't be calculated because the sample rate
                #  for the event is under the nyquist threshold, so mark this movement as noise since it's unknown
                if np.isnan(calculus_error):
                    event = 'noise'
                else:
                    event = prev_row.event
                
                # carpenter error
                carpenter_error = np.abs((duration*1000 - (21 + 2.2 * amplitude)) / (21 + 2.2 * amplitude))
                
                # include average probabilities if available
                try:
                    max_P_nonfix = np.nanmax([r.P_nonfix for r in stream])
                    P_fix = np.nansum([r.P_fix for r in stream])/len(stream)
                    P_nonfix = 1-P_fix
                    L_fix = np.nansum([r.L_fix for r in stream])/len(stream)
                    L_nonfix = np.nansum([r.L_nonfix for r in stream])/len(stream)
                except:
                    max_P_nonfix = np.nan
                    P_fix = np.nan
                    P_nonfix = np.nan
                    L_fix = np.nan
                    L_nonfix = np.nan
                try:
                    P_ff = np.nansum([r.P_ff for r in stream])/len(stream)
                    P_smp = np.nansum([r.P_smp for r in stream])/len(stream)
                    P_sac = np.nansum([r.P_sac for r in stream])/len(stream)
                    P_blink = np.nansum([r.P_blink for r in stream])/len(stream)
                except:
                    P_ff = np.nan
                    P_smp = np.nan
                    P_sac = np.nan
                    P_blink = np.nan
                try:
                    L_ff = np.nansum([r.L_ff for r in stream])/len(stream)
                    L_smp = np.nansum([r.L_smp for r in stream])/len(stream)
                    L_sac = np.nansum([r.L_sac for r in stream])/len(stream)
                    L_blink = np.nansum([r.L_blink for r in stream])/len(stream)
                except:
                    L_ff = np.nan
                    L_smp = np.nan
                    L_sac = np.nan
                    L_blink = np.nan
                    
                # flag if there's a blink
                try:
                    has_blink = int(any([r.has_blink for r in stream]))
                except:
                    has_blink = 0
                    
                # create event
                eventStream.append({'event':event,
                                    'start_i':stream_i[0],
                                    'end_i':stream_i[-1],
                                    'start_s':t[0],
                                    'end_s':t[-1],
                                    'duration':duration,
                                    'amplitude':amplitude,
                                    'dispersion':dispersion,
                                    'center_x':np.round(med_x,2),
                                    'center_y':np.round(med_y,2),
                                    'x0':x[0],
                                    'y0':y[0],
                                    'xn':x[-1],
                                    'yn':y[-1],
                                    'std_x':std_x,
                                    'std_y':std_y,
                                    'cov_x':cov_x,
                                    'cov_y':cov_y,
                                    'cov_xy':cov_xy,
                                    'max_Dn':max_Dn,
                                    'avg_Dn':avg_Dn,
                                    'max_iss':max_iss,
                                    'avg_iss':avg_iss,
                                    'max_vel':max_vel,
                                    'avg_vel':avg_vel,
                                    'std_vel':std_vel,
                                    'P_fix':P_fix,
                                    'P_nonfix':P_nonfix,
                                    'max_P_nonfix':max_P_nonfix,
                                    'L_fix':L_fix,
                                    'L_nonfix':L_nonfix,
                                    'P_ff':P_ff,
                                    'P_smp':P_smp,
                                    'P_sac':P_sac,
                                    'P_blink':P_blink,
                                    'L_ff':L_ff,
                                    'L_smp':L_smp,
                                    'L_sac':L_sac,
                                    'L_blink':L_blink,
                                    'calculus_error':calculus_error,
                                    'carpenter_error':carpenter_error,
                                    'has_blink':has_blink})
                prev_row = row
                prev_idx = idx
                stream = [prev_row]
                stream_i = [prev_idx]
                
    return pd.DataFrame(eventStream)


def merge_events(df, test_events_df, prev_i, next_i, event, interp_v, SAMPLE_RATE, blink=False):
    """
    A function that merges two events in an events dataframe.
    """
    prev_row = test_events_df.loc[prev_i]
    next_row = test_events_df.loc[next_i]
    
#     if event == 'fix':
#         # calculate amplitude as the average of the two fixation amplitudes (spread in dispersion)
#         amplitude = np.average([prev_row.amplitude, next_row.amplitude])     
#     else:
#         # calculate amplitude as the spatial distance between last and first of samples
#         amplitude = np.sqrt((next_row.xn-prev_row.x0)**2+(next_row.yn-prev_row.y0)**2)
    
    # find associated x and y positions in order to calculate features
    stream = pd.concat([df.loc[prev_row.start_i:prev_row.end_i], df.loc[next_row.start_i:next_row.end_i]], ignore_index=True)
    
    # ignore NAN values when calculating features    
    x = [r.x_deg for i, r in stream.iterrows() if not np.isnan(r.x_deg)]
    y = [r.y_deg for i, r in stream.iterrows() if not np.isnan(r.y_deg)]
    t = [r.time for i, r in stream.iterrows()]
    
    # if the event is non-fixation then add neighboring samples to x, y and t
    if event != 'fix' and prev_row.start_i != 0 and next_row.end_i < len(df)-1:
        prev_end = df.loc[prev_row.start_i-1]
        next_start = df.loc[next_row.end_i+1]
        x.append(next_start.x_deg)
        y.append(next_start.y_deg)
        t.append(next_start.time)
        x.insert(0, prev_end.x_deg)
        y.insert(0, prev_end.y_deg)
        t.insert(0, prev_end.time)
    
    # ignore NAN values when calculating features    
    x = [x_i for x_i in x if not np.isnan(x_i)]
    y = [y_i for y_i in y if not np.isnan(y_i)]
    
    # if there all values are nan then the event will have nans and be labelled as noise if it's not a blink
    if len(x) == 0 or len(y) == 0:
        x.append(np.nan)
        y.append(np.nan)
        if event != 'blink':
            event = 'noise'
    
    # calculate merged features
    cov_x, cov_y, cov_xy = np.cov(x), np.cov(y), np.cov(x,y)[0][1]
    std_x, std_y = np.std(x), np.std(y)
    med_x = median(x)
    med_y = median(y)
    start_i, start_s = prev_row.start_i, t[0]
    end_i, end_s = next_row.end_i, t[-1]
    duration = t[-1] - t[0]
    amplitude = np.sqrt((x[-1]-x[0])**2 + (y[-1]-y[0])**2)
    dispersion = std_x + std_y
    center_x = np.round(med_x,2)
    center_y = np.round(med_y,2)
    x0, y0, xn, yn = x[0], y[0], x[-1], y[-1]
    carpenter_error = np.abs((duration*1000 - (21 + 2.2 * amplitude)) / (21 + 2.2 * amplitude))
    integral = interp_v.integral(t[0],t[-1])
    calculus_error = np.abs((amplitude-integral)/integral)
    max_Dn = np.nanmax([r.Dn for i, r in stream.iterrows()])
    avg_Dn = np.nanmean([r.Dn for i, r in stream.iterrows()])
    max_iss = np.nanmax([r.iss for i, r in stream.iterrows()])
    avg_iss = np.nanmean([r.iss for i, r in stream.iterrows()])
    max_vel, avg_vel, std_vel = interpolate_velocity(t, interp_v, SAMPLE_RATE)
    
    # include average probabilities if available
    try:
        max_P_nonfix = np.nanmax([r.P_nonfix for r in stream])
        P_fix = np.nansum([r.P_fix for r in stream])/len(stream)
        P_nonfix = 1-P_fix
        L_fix = np.nansum([r.L_fix for r in stream])/len(stream)
        L_nonfix = np.nansum([r.L_nonfix for r in stream])/len(stream)
    except:
        max_P_nonfix = np.nan
        P_fix = np.nan
        P_nonfix = np.nan
        L_fix = np.nan
        L_nonfix = np.nan
    try:
        P_ff = np.nansum([r.P_ff for r in stream])/len(stream)
        P_smp = np.nansum([r.P_smp for r in stream])/len(stream)
        P_sac = np.nansum([r.P_sac for r in stream])/len(stream)
        P_blink = np.nansum([r.P_blink for r in stream])/len(stream)
    except:
        P_ff = np.nan
        P_smp = np.nan
        P_sac = np.nan
        P_blink = np.nan
    try:
        L_ff = np.nansum([r.L_ff for r in stream])/len(stream)
        L_smp = np.nansum([r.L_smp for r in stream])/len(stream)
        L_sac = np.nansum([r.L_sac for r in stream])/len(stream)
        L_blink = np.nansum([r.L_blink for r in stream])/len(stream)
    except:
        L_ff = np.nan
        L_smp = np.nan
        L_sac = np.nan
        L_blink = np.nan

    # set the previous event as the merged event with the updated features
    test_events_df.loc[prev_i] = {'event':event,
                                'start_i':start_i,
                                'end_i':end_i,
                                'start_s':start_s,
                                'end_s':end_s,
                                'duration':duration,
                                'amplitude':amplitude,
                                'dispersion':dispersion,
                                'center_x':np.round(center_x,2),
                                'center_y':np.round(center_y,2),
                                'x0':x0,
                                'y0':y0,
                                'xn':xn,
                                'yn':yn,
                                'std_x':std_x,
                                'std_y':std_y,
                                'cov_x':cov_x,
                                'cov_y':cov_y,
                                'cov_xy':cov_xy,
                                'max_Dn':max_Dn,
                                'avg_Dn':avg_Dn,
                                'max_iss':max_iss,
                                'avg_iss':avg_iss,
                                'max_vel':max_vel,
                                'avg_vel':avg_vel,
                                'std_vel':std_vel,
                                'P_fix':P_fix,
                                'P_nonfix':P_nonfix,
                                'max_P_nonfix':max_P_nonfix,
                                'L_fix':L_fix,
                                'L_nonfix':L_nonfix,
                                'P_ff':P_ff,
                                'P_smp':P_smp,
                                'P_sac':P_sac,
                                'P_blink':P_blink,
                                'L_ff':L_ff,
                                'L_smp':L_smp,
                                'L_sac':L_sac,
                                'L_blink':L_blink,
                                'calculus_error':calculus_error,
                                'carpenter_error':carpenter_error,
                                'has_blink':int(blink)}
    
    return test_events_df


# def adjust_endpoints(df, temp):
#     for idx, row in temp.iterrows():
#         temp.loc[idx, 'x0'] = df.loc[row.start_i,'x_deg']
#         temp.loc[idx, 'y0'] = df.loc[row.start_i,'y_deg']
#         temp.loc[idx, 'xn'] = df.loc[row.end_i,'x_deg']
#         temp.loc[idx, 'yn'] = df.loc[row.end_i,'y_deg']
#     return temp


def adjust_amp_dur(events_df, df, interp_v, first_run=False):
    """
    Adjusts amplitudes of non-fixations to be the euclidean distance from the prev fixation center to the center
    of the next fixation. Also adjusts durations to be time from end of prev event to start of next event.
    """
#     # first adjust endpoints to ensure they align with df
#     events_df = adjust_endpoints(df, events_df)
    
    for idx, row in events_df.iterrows():
        if idx == 0 or idx == len(events_df)-1:
            continue

        prev_row = events_df.loc[idx-1]
        next_row = events_df.loc[idx+1]
            
        # for all non-fixation events that occur between two fixations
        if row.event != 'fix' and prev_row.event == 'fix' and next_row.event == 'fix':
            if row.event == 'noise' or row.event == 'loss':
                continue
            A = np.sqrt((next_row.center_x-prev_row.center_x)**2+(next_row.center_y-prev_row.center_y)**2)
            D = row.end_s - row.start_s
            events_df.loc[idx,'amplitude'] = A
            events_df.loc[idx,'carpenter_error'] = np.abs((D*1000 - (21 + 2.2 * A)) / (21 + 2.2 * A))
            integral = interp_v.integral(row.start_s, row.end_s)
            calculus_error = np.abs((A-integral)/integral)
        
    return events_df


def map_to_stream(seq, df):
    df['event'] = np.where(df.event!='loss','other','loss')
    for i, row in seq.iterrows():
        df.loc[row.start_i:row.end_i, 'event'] = row.event
        # include likelihoods and probabilities if possible
        try:
            df['P_ff'] = row.P_ff
            df['P_smp'] = row.P_smp
            df['P_sac'] = row.P_sac
            df['P_blink'] = row.P_blink
        except:
            pass
        try:
            df['L_ff'] = row.L_ff
            df['L_smp'] = row.L_smp
            df['L_sac'] = row.L_sac
            df['L_blink'] = row.L_blink
        except:
            pass
    return df