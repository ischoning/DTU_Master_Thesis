"""
Functions used in plotting for the analysis of event detection.
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
from matplotlib.patches import Ellipse
import matplotlib.patches as mpatches
import matplotlib.transforms as transforms

#
# DO NOT TOUCH. Define colors and states for plotting events.
#
STATES = ['fix', 'sac', 'smp', 'vor', 'blink', 'fix_blink', 'noise', 'other', 'loss']
COLORS = {'fix':'red', 'sac':'lime', 'smp':'purple', 'vor':'brown', 'blink':'dodgerblue', 'fix_blink':'darkorchid', 'noise':'turquoise', 'other':'grey', 'loss':'gold'}
BORDERS = {'fix':'magenta', 'sac':'green', 'smp':'blue', 'vor':'tan', 'blink':'royalblue', 'fix_blink':'darkviolet', 'noise':'lightseagreen', 'other':'black', 'loss':'yellow'}
BINS = 50

def makePlot(df, show_events=[], title=''):
    if len(show_events) == 0:
        show_events = df.event.unique()
        show_events = [event for event in show_events if event != None]
    
        
    fig, ax = plt.subplots(figsize=(9,5))
    ax2 = ax.twinx()  # twin object for two different y-axes on the same plot
    
    # Plot position.
    ax2.plot(df.time, df.x_deg, label='x', alpha = 0.7, linewidth=0.6)
    ax2.plot(df.time, df.y_deg, label='y', alpha = 0.7, linewidth=0.6)
    ax2.set_ylabel('deg', color='orange', fontsize=10)
    
    # Plot speed.
    ax.plot(df.time, df.Dn, '.--', color='grey', alpha=0.8, label='Dn')
    ax.plot(df.time, df.Dy, '.--', color='tan', linewidth=0.8, alpha = 0.8, label='Dy')
    ax.plot([df.loc[0,'time'],df.loc[len(df)-1,'time']], [0,0], color='black', label='baseline Dy=0')
    ax.set_ylabel('Intersample Speed ($\degree s^{-1}$)', fontsize=10)
    ax.set_xlabel('Time (s)')
    
    # Plot each event a separate color.
    upper_bound = min(df[~np.isnan(df.iss)].iss)-20
    lower_bound = upper_bound-20    
    if len(show_events) > 0:
        for event in show_events:
            ax.plot(df.time, np.where(df['event']==event,0,None), '+',
                    color=COLORS[event], alpha=0.5, label=event)
            ax.fill_between(df.time, lower_bound, upper_bound, where= df.event == event,
                            color=COLORS[event], alpha=0.5)
    
    # labels
    ax.legend(loc='upper left')
    ax2.legend(loc='upper right')
    
    ax.set_title('Intersample Speed vs Time')
    
    lower = min(min(df[~np.isnan(df.x_deg)].x_deg), min(df[~np.isnan(df.y_deg)].y_deg)) - 5
    upper = max(max(df[~np.isnan(df.x_deg)].x_deg), max(df[~np.isnan(df.y_deg)].y_deg)) + 5
    
    ax2.set_yticks([i for i in range(int(lower), int(upper), 2)])
    ax2.grid(linewidth=0.5)
    
    plt.show()
    

def makePlot2(df, show_events, show_events_bool=False):
    fig, ax = plt.subplots(2,1, sharex=True, figsize=(10,5))


    # Plot position.
    ax[1].plot(df.time, df.y_deg, color='orange', label='y', alpha = 0.7, linewidth=3)
    ax[1].plot(df.time, df.x_deg, '--', color='orange', label='x', alpha = 0.7, linewidth=3)
    ax[1].set_ylabel('position ($\degree$)', fontsize=15)#color='orange', fontsize=10)
    ax[1].axis()

    # Plot speed.
    try:
        ax[0].plot(df.time, df.Dn, '.--', linewidth=2, alpha=0.8, label='Dn')
    except:
        pass
    #ax[0].plot(df.time, df.Dy, linewidth=3, alpha = 0.8, label="y'(t)")
    ax[0].plot(df.time, df.iss, linewidth=2, alpha = 0.8, label="iss")
    #ax.plot([df.loc[0,'time'],df.loc[len(df)-1,'time']], [0,0], color='black', label='baseline Dy=0')
    ax[0].set_ylabel("velocity ($\degree s^{-1}$)", fontsize=15)
    ax[1].set_xlabel('time (sec)',fontsize=15)

    if show_events_bool:
        # Plot each event a separate color.
        upper_bound = np.nanmin(df.iss)-20
        lower_bound = upper_bound-20    
        if len(show_events) > 0:
            for event in show_events:
                ax[0].plot(df.time, np.where(df['event']==event,0,None), '+',
                        color=COLORS[event], alpha=0.5, label=event)
                ax[0].fill_between(df.time, lower_bound, upper_bound, where= df.event == event,
                                color=COLORS[event], alpha=0.5)

    # labels
    ax[0].legend(bbox_to_anchor=(-.05,1,1,0), loc='lower left',ncol=len(show_events)+2,fontsize=12)
    ax[1].legend(fontsize=12)

    #ax.set_title('Intersample Speed vs Time')

    lower = min(df[~np.isnan(df.iss)].iss) - 5
    upper = max(df[~np.isnan(df.iss)].iss) + 5
    ax[0].set_yticks([i for i in range(-1000, 1500, 250)])
    ax[0].grid(linewidth=0.5)
    ax[0].set_ylim((-1000,1500))

#     lower = min(min(df[~np.isnan(df.x_deg)].x_deg), min(df[~np.isnan(df.y_deg)].y_deg)) - 5
#     upper = max(max(df[~np.isnan(df.x_deg)].x_deg), max(df[~np.isnan(df.y_deg)].y_deg)) + 5
    lower = min(np.nanmin(df.x_deg), np.nanmin(df.y_deg)) - 5
    upper = max(np.nanmax(df.x_deg), np.nanmax(df.y_deg)) + 5
    ax[1].set_yticks([i for i in range(-100, 100, 10)])
    ax[1].grid(linewidth=0.5)

    plt.rc('xtick', labelsize=15)
    plt.rc('ytick', labelsize=15)

    plt.show()
    
def feature_boxplot_by_event(seq, feat, q=1):
    """
    Makes boxplot comparison by event for specified feature and quantile q.
    """
    plt.figure(figsize=(10,5))
    events = seq.loc[(seq.event!='noise')&(seq.event!='loss')].event.unique().tolist()
    data = []
    for e in events:
        data.append((seq.loc[(seq.event == e)&(seq[feat] < seq[feat].quantile(q)), feat]))
    plt.boxplot(data, labels = events)
#     plt.title(f'Simple boxplot of {feat} by event (q={q})', fontsize=15)
    plt.title(f'{feat} by event (q={q})', fontsize=15)
    plt.ylabel(feat, fontsize=15)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.show()
    
    
def feature_histogram(seq, feat, event, q=1,x=None):
    """
    Makes simple boxplot for event feature with quantile q.
    """
    e = {'sac':'Saccade', 'smp':'Smooth Pursuit', 'fix':'Fixation', 'blink':'Blink', 'noise':'Noise', 'other':'Other'}
    q = 1
    
    plt.figure(figsize=(6,5))
    temp = seq.loc[(seq.event == event)&(seq[feat] < seq[feat].quantile(q)), feat]        
    if (feat == 'duration' and e != 'blink') or (feat == 'dispersion' and e == 'sac'):
        if event == 'sac' and feat == 'duration':
            bins = 200
        else:
            bins = 100
    else:
        if len(temp)//2 < 50:
            bins = len(temp)//2
        else:
            bins = 30
    plt.hist(temp, 
             bins=bins, 
             label = f'mean={np.round(np.average(temp),2)}\nstd={np.round(np.std(temp),2)}\nN={len(temp)}')
    #plt.title(f'Simple histogram of {feat}: {e[event]} (q={q})', fontsize=15)
    plt.title(f'{feat}: {e[event]}', fontsize=15)
    plt.xlabel(feat, fontsize=15)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.legend(fontsize=15)
    if x is not None:
        plt.xlim((0,x))
    plt.show()
    
    
def plot_change(seqs, event, q=1):
    """
    Makes boxplot of sequences seqs for comparison of features for "event" where q is the quantile to plot.
    """
    
    e = {'sac':'Saccade', 'smp':'Smooth Pursuit', 'fix':'Fixation', 'blink':'Blink'}
    
    if len(seqs) == 2:
        labels = ['before', 'after']
    else:
        labels = [f'step {i+1}' for i in range(len(seqs))]

    fig, ax = plt.subplots(2,2,figsize=(10,10))
    
    plt.suptitle(f'{e[event]} (q={q})', fontsize=20)

    data = []
    for seq in seqs:
        data.append((seq[seq.event == event][seq.duration < seq.duration.quantile(q)].duration))
    ax[0][0].boxplot(data, labels = ['' for i in range(len(seqs))])
    ax[0][0].set_title('Duration', fontsize=20)
    ax[0][0].set_ylabel('Duration (s)', fontsize=20)
    
    data = []
    for seq in seqs:
#         if event == 'fix':
#             data.append((seq[seq.event == event][seq.dispersion < seq.dispersion.quantile(q)].dispersion))
#         else:
        data.append((seq[seq.event == event][seq.amplitude < seq.amplitude.quantile(q)].amplitude))
    ax[0][1].boxplot(data, labels = ['' for i in range(len(seqs))])
    if event == 'fix':
        t = 'Dispersion'
    else: t = 'Amplitude'
    ax[0][1].set_title(t, fontsize=20)
    ax[0][1].set_ylabel(f'{t} ($\degree$)', fontsize=20)
    ax[0][1].yaxis.set_label_position('right')
    ax[0][1].yaxis.tick_right()

    data = []
    for seq in seqs:
        data.append((seq[seq.event == event][seq.avg_iss < seq.avg_iss.quantile(q)].avg_iss))
    ax[1][0].boxplot(data, labels = labels)
    ax[1][0].set_title('Avg Speed', fontsize=20)
    ax[1][0].set_ylabel('Speed ($\degree s^{-1}$)', fontsize=20)
    
    data = []
    for seq in seqs:
        data.append((seq[seq.event == event][seq.max_iss < seq.max_iss.quantile(q)].max_iss))
    ax[1][1].boxplot(data, labels = labels)
    ax[1][1].set_title('Peak Speed', fontsize=20)
    ax[1][1].set_ylabel('Speed ($\degree s^{-1}$)', fontsize=20)
    ax[1][1].yaxis.set_label_position('right')
    ax[1][1].yaxis.tick_right()
    
#     data = []
#     for seq in seqs:
#         data.append((seq[seq.event == event][seq.carpenter_error < seq.carpenter_error.quantile(q)].carpenter_error))
#     ax[1][1].boxplot(data, labels = labels)
#     ax[1][1].set_title('Carpenter Error', fontsize=15)
#     ax[1][1].set_ylabel('Error', fontsize=15)
#     ax[1][1].yaxis.set_label_position('right')
#     ax[1][1].yaxis.tick_right()

    plt.rc('xtick', labelsize=20)
    plt.rc('ytick', labelsize=20)
    
    plt.show()
    
    

def plot_hists(seq,q=1):
    states = ['fix', 'smp', 'sac']
    feats = ['amplitude', 'dispersion']
    titles = {'fix':'Fixation','smp':'Smooth pursuit','sac':'Saccade'}
    labels = {'amplitude':'Amplitude ($\degree$)', 'dispersion':'Dispersion ($\degree$)'}

    fig, ax = plt.subplots(2,3,figsize=(10,10))

    plt.suptitle(f'q={q}', fontsize=15)

    for i in range(len(feats)):
        feat = feats[i]
        for j in range(len(states)):
            e = states[j]
            y, x, _ = ax[i][j].hist(seq.loc[(seq.event==e)&(seq[feat] < seq[feat].quantile(q)),feat],bins=20)
            ax[i][j].vlines(x=1,ymin=0,ymax=y.max(),color='orange',linewidth=3,linestyle='--')
            if j == 0:
                ax[i][j].set_ylabel('Frequency', fontsize=20)
            if j == 1:
                ax[i][j].set_xlabel(labels[feat], fontsize=20)
            ax[i][j].set_title(titles[e],fontsize=20)
    plt.rc('xtick', labelsize=20)
    plt.rc('ytick', labelsize=20)
    plt.subplots_adjust(hspace=0.4)
    plt.show()

    

def GMM_plot(df, feat, params):
    mu1, mu2, sd1, sd2, w1, w2 = params
    if feat == 'Dn':
        label1 = '$D_n$'
        label2 = r'$\mathcal{N}(L)$,'
        label3 = r'$\mathcal{N}(H)$,'
        units = 'deg/sec'
    
    fig, ax = plt.subplots(1,2, figsize=(15,6))

    x_min = min(df[df.feat<df.feat.quantile(.95)].feat)
    x_max = max(df[df.feat<df.feat.quantile(.95)].feat)

    sns.distplot(df.feat, bins = 500, kde_kws=dict(linewidth=4), label = f'{label1}', ax=ax[0])
    x_axis = np.linspace(mu1-3*sd1, mu2+3*sd2, 1000)
    ax[0].plot(x_axis, norm.pdf(x_axis, mu1, sd1), linewidth=4, label = label2+f' $\mu_1={int(mu1)}, \sigma_1={int(sd1)}, w_1={np.round(w1,2)}$')
    ax[0].plot(x_axis, norm.pdf(x_axis, mu2, sd2), linewidth=4, label = label3+f' $\mu_2={int(mu2)}, \sigma_2={int(sd2)}, w_2={np.round(w2,2)}$')
    ax[0].legend(bbox_to_anchor=(-0.05,-0.5,1.05,0), loc='lower center', mode='expand',fontsize=18)
    ax[0].set_title('Distribution of $D_n$',fontsize=20)
    ax[0].set_xlabel(label1+' '+units,fontsize=20)
    ax[0].set_ylabel('Probability',fontsize=20)
    ax[0].set_xlim(x_min,x_max)

    ax[1].scatter(df.Dn, df.L_fix, color='orange', label='$p(D_n|L)$')
    ax[1].scatter(df.Dn, df.L_nonfix, color='green', label='$p(D_n|H)$')
    ax[1].legend(bbox_to_anchor=(0.3,-0.4,.5,0), loc='lower center', mode='expand',fontsize=18)
    ax[1].set_xlabel(label1+' '+units,fontsize=20)
    ax[1].set_ylabel('likelihood',fontsize=20)
    ax[1].set_title('Likelihood in '+label1,fontsize=20)
    ax[1].set_xlim(x_min,x_max)
    ax[1].yaxis.set_label_position('right')
    ax[1].yaxis.tick_right()

    plt.rc('xtick', labelsize=15)
    plt.rc('ytick', labelsize=15)

    plt.show()
    
    
def Dy_plot(df):

    fig, ax = plt.subplots(2,2, figsize=(9,5))

    #plt.autoscale(False)
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)

    temp = df.loc[(df.time > 98.46) & (df.time < 98.61)]
    temp2 = df.loc[(df.time > 98.65) & (df.time < 99)]

    # Plot position.
    ax[1][0].plot(temp.time, temp.y_deg, color='orange', label='y', alpha = 0.7, linewidth=2)
    ax[1][0].plot(temp.time, temp.x_deg, '--', color='orange', label='x', alpha = 0.7, linewidth=2)
    ax[1][0].set_ylabel('x, y ($\degree$)', fontsize=15)#color='orange', fontsize=10)
    ax[1][0].axis()
    ax[1][0].set_yticks([i for i in range(-50, 50, 10)])
    ax[1][0].grid(linewidth=0.5)
    ax[1][0].set_ylim((min(min(temp.x_deg), min(temp.y_deg))-5,max(max(temp.x_deg), max(temp.y_deg))+5))
    ax[1][0].set_xlim((min(temp.time),max(temp.time)))
    ax[1][0].set_xlabel("time (sec)", fontsize=15)

    #^
    # Plot speed.
    #ax.plot(df.time, df.Dn, '.--', color='grey', alpha=0.8, label='Dn')
    ax[0][0].plot(temp.time, temp.Dy, linewidth=3, alpha = 0.8, label="y'(t)")
    #ax.plot([df.loc[0,'time'],df.loc[len(df)-1,'time']], [0,0], color='black', label='baseline Dy=0')
    ax[0][0].set_ylabel("y'(t) ($\degree s^{-1}$)", fontsize=15)

    lower = min(temp[~np.isnan(temp.Dy)].Dy) - 5
    upper = max(temp[~np.isnan(temp.Dy)].Dy) + 5
    ax[0][0].set_yticks([i for i in range(-1000, 1000, 250)])
    ax[0][0].grid(linewidth=0.5)
    ax[0][0].set_ylim((min(temp.Dy)-50,max(temp.Dy)+50))
    ax[0][0].set_xlim((min(temp.time),max(temp.time)))

    maxi = np.argmax(np.array(temp.Dy))
    mini = np.argmin(np.array(temp.Dy))
    x_max, x_min = np.array(temp.time)[maxi], np.array(temp.time)[mini]
    y_max, y_min = np.array(temp.Dy)[maxi], np.array(temp.Dy)[mini]


    ax[0][0].annotate('max'+r'$\approx$ 280', xy=(x_max,y_max), xytext=(x_max+0.005,y_max-20), fontsize=12)
    ax[0][0].annotate('min'+r'$\approx$ -100', xy=(x_min,y_min), xytext=(x_min+0.005,y_min), fontsize=12)
    ax[0][0].scatter((x_max),(y_max))
    ax[0][0].scatter((x_min),(y_min))

    # Plot position.
    ax[1][1].plot(temp2.time, temp2.y_deg, color='orange', label='y', alpha = 0.7, linewidth=2)
    ax[1][1].plot(temp2.time, temp2.x_deg, '--', color='orange', label='x', alpha = 0.7, linewidth=2)
    ax[1][1].set_yticks([i for i in range(-50, 50, 10)])
    ax[1][1].grid(linewidth=0.5)
    # labels
    ax[1][1].legend(bbox_to_anchor=(1,0.5), fontsize=12)
    ax[1][1].set_xlabel('time (sec)',fontsize=15)
    ax[1][1].set_ylim((min(min(temp2.x_deg), min(temp2.y_deg))-5,max(max(temp2.x_deg), max(temp2.y_deg))+5))
    ax[1][1].set_xlim((min(temp2.time),max(temp2.time)))


    # Plot speed.
    #ax.plot(df.time, df.Dn, '.--', color='grey', alpha=0.8, label='Dn')
    ax[0][1].plot(temp2.time, temp2.Dy, linewidth=3, alpha = 0.8, label="y'(t)")
    #ax.plot([df.loc[0,'time'],df.loc[len(df)-1,'time']], [0,0], color='black', label='baseline Dy=0')
    maxi = np.argmax(np.array(temp2.Dy))
    mini = np.argmin(np.array(temp2.Dy))
    x_max, x_min = np.array(temp2.time)[maxi], np.array(temp2.time)[mini]
    y_max, y_min = np.array(temp2.Dy)[maxi], np.array(temp2.Dy)[mini]


    ax[0][1].annotate('max'+r'$\approx$ 750', xy=(x_max,y_max), xytext=(x_max+0.005,y_max-100), fontsize=12)
    ax[0][1].annotate('min'+r'$\approx$ -300', xy=(x_min,y_min), xytext=(x_min+0.02,y_min), fontsize=12)

    ax[0][1].scatter((x_max),(y_max))
    ax[0][1].scatter((x_min),(y_min))

    ax[0][1].set_yticks([i for i in range(-1000, 1000, 250)])
    ax[0][1].grid(linewidth=0.5)
    ax[0][1].set_ylim((min(temp2.Dy)-50,max(temp2.Dy)+50))
    ax[0][1].set_xlim((min(temp2.time),max(temp2.time)))
    #ax[0][1].legend(bbox_to_anchor=(1,0.5), fontsize=15)

    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)

    plt.show()


def plot_carpenter(seqDF, show_events=[], q=1):
    
    if len(show_events) == 0:
        show_events = seqDF.event.unique()
        show_events = [event for event in show_events if event != None]
        
    # plot Carpenter's Theorem
    plt.figure(figsize=(12,9))
    xmin = min(seqDF[seqDF.event=='sac'].amplitude)
    xmax = max(seqDF[seqDF.event=='sac'].amplitude)
    x = np.linspace(xmin, xmax)
    y = 21 + 2.2*x
    plt.plot(x, y, color = 'black', linewidth = 1, label = 'D = 21 + 2.2A')
    
    plt.vlines(x=1,ymin=-100,ymax=4000,linestyle='--',color='black',linewidth=1,label=r'1$\degree$')

#     # plot error bounds (20%)
#     y_u = 21*(1+carp_thresh) + 2.2*x
#     y_l = 21*(1-carp_thresh) + 2.2*x
#     plt.plot(x, y_u, color = 'black', linestyle = 'dashed', linewidth = 0.7)
#     plt.plot(x, y_l, color = 'black', linestyle = 'dashed', linewidth = 0.7, label = 'error bound')

    # plot events
    for event in show_events:
        seq = seqDF[seqDF['event']==event]    # convert duration from s to ms
        if event == 'sac':
            seq = seqDF.loc[(seq.amplitude > 1.5)&(seqDF['event']==event)]
        plt.scatter(seq.loc[seq['amplitude'] < seq['amplitude'].quantile(q), 'amplitude'], 
                    seq.loc[seq['amplitude'] < seq['amplitude'].quantile(q), 'duration']*1000,
                    color = COLORS[event],
                    edgecolor=BORDERS[event],
                    label = event,
                    alpha = 0.5)#,
                    #s =2)
    
    plt.xlabel('Amplitude ($\degree$)',fontsize=20)
    plt.ylabel('Duration (ms)',fontsize=20)
    plt.title(f'Amplitude vs Duration of Events',fontsize=20)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.ylim(-100,4000)
    plt.xlim((-2,80))
    plt.legend(loc = 'upper right',fontsize=20)
    #plt.legend(bbox_to_anchor=(1.02,1),loc = 'upper left',fontsize=15)
    #plt.hlines(y=100,xmin=xmin,xmax=xmax,linestyle='--',color='black',linewidth=1)
    plt.show()
    
def plot_interp_vel(t, interp_dx, interp_dy, SAMPLE_RATE, title=''):
    """
    Returns the peak, avg, and std of intersample velocity of upsampled interpolated data over a given time.
    """  
    # Upsample features
    interp_t = np.linspace(t[0], t[-1], num=np.ceil(500/SAMPLE_RATE).astype(int)*len(t))
    vels = [interp_dx(t_i)+interp_dy(t_i) for t_i in interp_t]
    y_vels = interp_dy(interp_t)
    
    # Calculate features (ignore nans)
    max_vel = int(np.nanmax(vels))
    avg_vel = int(np.nanmean(vels))
    std_vel = int(np.nanstd(vels))
    
    # Return max and min velocity in y-direction too, for use in Dv calculation
    max_dy, min_dy = int(np.nanmax(y_vels)), int(np.nanmin(y_vels))
    
    plt.figure()
    plt.plot(interp_t, vels, linewidth=3, label='y_vel')
    plt.plot(interp_t, y_vels, linewidth=3, label='Dy')
    plt.legend(fontsize=20)
    plt.title(f'Interpolated Velocities ({title})',fontsize=20)
    plt.xlabel('time (s)',fontsize=20)
    plt.ylabel('vel ($\degree s^{-1}$)',fontsize=20)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.show()
    print(f"max_vel: {max_vel}\navg_vel: {avg_vel}\nstd_vel: {std_vel}\nmax_dy: {max_dy}\nmin_dy: {min_dy}")
    
def confidence_ellipse(x,y,ax,n_std=3.0,edgecolor='red',facecolor='none'):
    cov = np.cov(x,y)
    pearson = cov[0,1]/np.sqrt(cov[0,0]*cov[1,1])
    # use special case to obtain eigenvalues
    ell_rad_x = np.sqrt(1+pearson)
    ell_rad_y = np.sqrt(1-pearson)
    ellipse = Ellipse((0,0), width=ell_rad_x*2, height=ell_rad_y*2, linestyle='--', edgecolor=edgecolor, 
                      linewidth=2, facecolor=facecolor, alpha = 0.5)
    
    scale_x = np.sqrt(cov[0,0])*n_std
    mean_x = np.mean(x)
    
    scale_y = np.sqrt(cov[1,1])*n_std
    mean_y = np.mean(y)
    
    transf = transforms.Affine2D() \
        .rotate_deg(45) \
        .scale(scale_x,scale_y) \
        .translate(mean_x,mean_y)
    
    ellipse.set_transform(transf + ax.transData)
    
    return ax.add_patch(ellipse)
    
    
def plot_events_scatter(temp, df, START, END):    
    fix, ax = plt.subplots(1,1,figsize=(12,12))
    start = temp[temp.start_s>df.loc[START,'time']].start_i.to_numpy()[0]
    end = temp[temp.end_s<df.loc[END,'time']].end_i.to_numpy()[-1]
    ax.scatter(df.x_deg[start:end], df.y_deg[start:end], label='original')
    i = 0
    for idx, row in temp.iterrows():
        e = row.event
        #ax.plot(df.loc[row.start_i:row.end_i,'x_deg'], df.loc[row.start_i:row.end_i,'y_deg'], color=COLORS[e], linewidth=2)
        if e == 'fix':
            #plt.gca().add_patch(Ellipse([row.center_x, row.center_y], width=row.std_x, height=row.std_y, color=COLORS[e], alpha=0.5))
            confidence_ellipse(df.loc[row.start_i:row.end_i,'x_deg'], df.loc[row.start_i:row.end_i,'y_deg'], ax, edgecolor=COLORS[e])
        else:
            ax.plot([row.x0,row.center_x,row.xn], [row.y0,row.center_y,row.yn], color=COLORS[e], linewidth=2)
        # set circle at center
        #ax.scatter([row.center_x], [row.center_y], s=100, color=COLORS[e], edgecolors='black', linewidth=2)
        ax.scatter([row.center_x], [row.center_y], color='black', edgecolor=COLORS[e], s=200)
        ax.scatter([row.center_x], [row.center_y], marker='x', s=100, color=COLORS[e],edgecolor='black', linewidth=2)
        ax.text(row.center_x+0.5, row.center_y-0.5, i, fontsize=25)
        i += 1
    ax.set_ylabel(r'y-position ($\degree$)', fontsize=25)
    ax.set_xlabel(r'x-position ($\degree$)', fontsize=25)
    ax.set_title('Gaze position', fontsize = 25)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    patches = [mpatches.Patch(color=COLORS[e], label=e) for e in temp.event.unique()]
    ax.legend(handles=patches, fontsize=20)
    plt.show()

    
    
def show_transition_matrix(temp, normalize=False):
    c = []
    c_norm = []
    for subID in temp.subID.unique():
        for task in temp.loc[temp.subID==subID, 'task'].unique():
            t = temp.loc[(temp.subID==subID) & (temp.task==task)]

            # rows are first event, columns are next event
            a = t[:len(t)-1].event.reset_index()
            a.rename(columns={'event':'event_i'},inplace=True)
            a = a['event_i']
            b = t[1:].event.reset_index()
            b.rename(columns={'event':'event_i+1'},inplace=True)
            b = b['event_i+1']
            c_ = pd.crosstab(a,b, margins=True)
            c_norm_ = pd.crosstab(a,b,normalize=1, margins=True)

            if 'loss' in c_.columns:
                c_ = c_.drop('loss', axis=0)
                c_ = c_.drop('loss', axis=1)
                c_norm_ = c_norm_.drop('loss', axis=0)
                c_norm_ = c_norm_.drop('loss', axis=1)
            if 'noise' in c_.columns:
                c_ = c_.drop('noise', axis=0)
                c_ = c_.drop('noise', axis=1)
                c_norm_ = c_norm_.drop('noise', axis=0)
                c_norm_ = c_norm_.drop('noise', axis=1)
            if 'blink' not in c_.columns:
                c_['blink'] = 0.0
                c_ = pd.concat([c_,pd.DataFrame(0.0,index=['blink'],columns=c_.columns)], axis=0)
                c_.index.name = 'event_i'
                c_norm_['blink'] = 0.0
                c_norm_ = pd.concat([c_norm_,pd.DataFrame(0.0,index=['blink'],columns=c_.columns)], axis=0)
                c_norm_.index.name = 'event_i'
                    
            if len(c) == 0:
                c.append(c_)
                c_norm.append(c_norm_)
            else:
                c.append(c_)
                c_norm.append(c_norm_)
    
    if not normalize:
        return np.round(sum(c)/len(c),2)
    else:
        return np.round(sum(c_norm)/len(c_norm),2)
    
    #return c, c_norm    
    
    
def show_linear_relation(seq,q=1):
    states = ['fix', 'smp', 'sac']
    feats = ['duration', 'avg_iss', 'max_iss']
    titles = {'fix':'Fixation','smp':'Smooth pursuit','sac':'Saccade'}
    labels = {'duration':'Duration (s)', 'avg_iss':'Avg speed ($\degree s^{-1}$)', 'max_iss':'Max speed ($\degree s^{-1}$)'}

    fig, ax = plt.subplots(3,3,figsize=(10,10))

    plt.suptitle(f'q={q}', fontsize=15)

    for j in range(len(states)):
        e = states[j]
        for i in range(len(feats)):
            y_feat = feats[i]
#             if e == 'sac':
            x_feat = 'amplitude'
            t = 'Amplitude ($\degree$)'
#             else:
#                 x_feat = 'dispersion'
#                 t = 'Dispersion ($\degree$)'
            temp = seq.loc[(seq.event==e)&(seq[x_feat] < seq[x_feat].quantile(q))]
            ax[i][j].scatter(temp[x_feat], temp[y_feat], s=15)
            ax[i][j].grid()
            # plot line of best fit?
            if i == 0:
                ax[i][j].set_title(titles[e],fontsize=15)
            if j == 0:
                ax[i][j].set_ylabel(labels[y_feat], fontsize=15)
            if i == 2:
                ax[i][j].set_xlabel(t, fontsize=15)

    plt.rc('xtick', labelsize=15)
    plt.rc('ytick', labelsize=15)
    plt.subplots_adjust(hspace=0.6)
    plt.grid()
    plt.show()

    
def analyze_datasets(d, title=''):
    """
    Plots the distribution of the values in dictionary d where keys are dataset IDs and values are datapoints.
    """
    
    number_dataset = {}
    i = 1
    for k in d.keys():
        number_dataset[k] = str(i)
        i += 1

    ds = [number_dataset[k] for k in d.keys()]

    plt.figure(figsize=(12,6))
    plt.bar(ds, d.values())
    plt.xlabel('Datasets', fontsize=18)
    plt.ylabel('Ratio', fontsize=18)
    plt.title(title, fontsize=18)
    plt.xticks(rotation=90, fontsize=0)
    plt.hlines(0.1,min([int(i) for i in ds])-2,max([int(i) for i in ds]), linestyle='dotted', linewidth=2, color='black',label='10%')
    plt.hlines(0.05,min([int(i) for i in ds])-2,max([int(i) for i in ds]), linestyle='--', linewidth=2, color='black',label='5%')
    plt.legend(fontsize=18)
    plt.yticks(fontsize=15)
    plt.show()
    
    
    
def show_correlation(seq, e, x_feat, q=1):
    feats = ['duration', 'avg_iss', 'max_iss']
    titles = {'fix':'Fixation','smp':'Smooth pursuit','sac':'Saccade'}
    labels = {'duration':'Duration (s)', 'avg_iss':'Avg speed ($\degree s^{-1}$)', 'max_iss':'Max speed ($\degree s^{-1}$)'}

    fig, ax = plt.subplots(1,3,figsize=(10,5))

    plt.suptitle(f'{titles[e]} (q={q})', fontsize=15)
    
    rs, ps = {}, {}
    for i in range(len(feats)):
        y_feat = feats[i]
        
        temp = seq.loc[(seq.event==e)&(seq[x_feat] < seq[x_feat].quantile(q))]
        
        temp = temp.loc[(~np.isnan(temp[x_feat])&(~np.isnan(temp[y_feat])))]
        
        x, y = temp[x_feat], temp[y_feat]
        
        # plot line of best fit
        a, b = np.polyfit(x,y,1)
        
        # correlation
        r, p = pearsonr(x,y)
        
        ax[i].scatter(temp[x_feat], temp[y_feat], s=15, label=f'r = {np.round(r,2)}\ny = {np.round(a,2)}*x+{np.round(b,2)}')
        ax[i].plot(temp[x_feat], a*temp[x_feat]+b, '--', color='black', linewidth= 1)
        
        ax[i].grid()
        
        ax[i].set_ylabel(labels[y_feat], fontsize=15)
        ax[i].legend(bbox_to_anchor=(0,1,1,0),loc='lower left',fontsize=15)

        ax[i].set_xlabel('Amplitude ($\degree$)')
        
        rs[y_feat] = r
        ps[y_feat] = p
        
    plt.rc('xtick', labelsize=15)
    plt.rc('ytick', labelsize=15)
    plt.subplots_adjust(hspace=0.6)
    plt.grid()
    plt.show()
    return rs, ps