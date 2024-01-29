#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 12 00:04:20 2023

@author: wuttinan
"""

import inspy as ins
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib
import os
import pickle
from visual import *
from temporal import *
from time_series import *
from extracted import displacement_least_square

data_path    = Path('./dataset')
dataset_name = Path('/Venu.LiCSBASv2.mod.xyz')

#ts_path = np.array(['right_asc', 'right_des', 'left_asc', 'left_des'])       # Directory to store timeseries dump.
ts_path = np.array(['right_asc'])

heading_angle = np.array([350,192,376,162])

# Load dataset.
with open(data_path / f'X.pkl','rb') as lat:
    X = pickle.load(lat)

with open(data_path / f'Y.pkl','rb') as lon:
    Y = pickle.load(lon)

with open(data_path / f'Vel_day.pkl','rb') as vel:
    V = pickle.load(vel)

with open(data_path / f'Coherent.pkl','rb') as coh:
    C = pickle.load(coh)

# seed = 15; given full covered temporal baselines
np.random.seed(15)

dstart = '20180214'
dstop  = '20200214'

u = np.array([0.0001, 0.0002, 0.0005, 0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1])
u_name = np.array(['1b-4', '2b-4', '5b-4', '1b-3', '2b-3', '5b-3', '1b-2', '2b-2', '5b-2', '1b-1', '2b-1', '5b-1', '1b0'])


for order, directory in enumerate(ts_path):
    for sub_order, regulate_params in enumerate(u):

        print('Initiated : Tikhonov regularisations (u): ', regulate_params)

        ifgs, num_pairs, cov_Lc = temporal_baselines_network(dstart,dstop,directory)

        params = {
            'heading'   :     np.deg2rad(heading_angle[order]),
            'look_angle':     np.deg2rad(34),
            'choice'    :     directory,
            'lon_mg'    :     X,
            'lat_mg'    :     Y,
            'vel_day'   :     V,
            'water_mask':     C,
            'tikhonov_ratio': regulate_params,
            'ifg_network' :   np.array(ifgs),      # [[ dstart, separation, time length ],[ dstop, separation, time length ]]
            'num_pairs' :     num_pairs,
            'ph_disturb':     0.02,     # Atmospheric disturbance in metres.
            'cov_Lc'    :     cov_Lc,   # Covariance length in metres.
            'dataset_directory' :    './dataset',
            'dump_directory'    :    directory  }

        tss = generate_time_series_network(params)                  # Generate the time series

        sbas = small_baseline_subset_inversion(params,tss)

        disps_tss, rms_before_LS = los_displacement(params,sbas,'before_LS_' + str(u_name[sub_order]))

        v_est = displacement_least_square(disps_tss)

        norder = len(rms_before_LS)                                 # assign the length of time frame.

        rms_after_LS = const_displacement(params,norder,v_est[0,:,:],'After_LS_' + str(u_name[sub_order]))

        x = np.arange(0,len(rms_after_LS),1) * 12
        fig,ax = plt.subplots()
        ax.scatter(x, rms_before_LS, c='b',label='residual before applying Least square to time series') # apply 2D inverse theory as part of an assumptions
        ax.scatter(x, rms_after_LS, c='r',label='residual after applying Least square to time series')
        ax.set_xlabel('Dates (days)')
        ax.set_ylabel('rms misfit (metres)')
        fig.savefig( Path(params['dataset_directory']) / Path(params['dump_directory']) / f'sbas_pic' / f'misfit_{str(u_name[sub_order])}')
        plt.close()


'''
for order, directory in enumerate(ts_path):

    print(order,'started',directory,'initialised')

    ifgs, num_pairs, cov_Lc = temporal_baselines_network(dstart,dstop,directory)

    params = {
        'heading'   :     np.deg2rad(heading_angle[order]),
        'look_angle':     np.deg2rad(34),
        'choice'    :     directory,
        'lon_mg'    :     X,
        'lat_mg'    :     Y,
        'vel_day'   :     V,
        'water_mask':     C,
        'tikhonov_ratio': u,
        'ifg_network' :   np.array(ifgs),      # [[ dstart, separation, time length ],[ dstop, separation, time length ]]
        'num_pairs' :     num_pairs,
        'ph_disturb':     0.02,     # Atmospheric disturbance in metres.
        'cov_Lc'    :     cov_Lc,   # Covariance length in metres.
        'dataset_directory' :    './dataset',
        'dump_directory'    :    directory}

    tss = generate_time_series_network(params)

    sbas = small_baseline_subset_inversion(params,tss)

    #visual_time_series(X,Y,sbas,C,params)

    disps_tss, rms_before_LS = los_displacement(params,sbas,'before_LS')

    v_est = displacement_least_square(disps_tss)

    norder = len(rms_before_LS)         # assign the length of time frame.

    rms_after_LS = const_displacement(params,norder,v_est[0,:,:],'After_LS')

    x = np.arange(0,len(rms_after_LS),1) * 12
    fig,ax = plt.subplots()
    ax.scatter(x, rms_before_LS, c='b',label='residual before applying Least square to time series') # apply 2D inverse theory as part of an assumptions
    ax.scatter(x, rms_after_LS, c='r',label='residual after applying Least square to time series')
    ax.set_xlabel('Dates (days)')
    ax.set_ylabel('rms misfit (metres)')
    plt.show()
'''


