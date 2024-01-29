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
from extracted import InSAR_extraction

data_path    = Path('./dataset')
dataset_name = Path('/Venu.LiCSBASv2.mod.xyz')

ts_path = np.array(['right_asc', 'right_des',
            'left_asc', 'left_des'])       # Directory to store timeseries dump.

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

tss = np.zeros([4,400,170])

dstart = '20160214'
dstop  = '20180214'
for order, directory in enumerate(ts_path):

    with open( Path('./dataset') / Path(directory) / Path('SBAS_time_series.pkl'), 'rb') as file:
        temp = pickle.load(file)
        tss[order,:,:] = temp[0,:,:]

ndim, nrow, ncol = tss.shape
vel = np.zeros([3,400,170])
print(tss.shape)

G = InSAR_extraction(ts_path,heading_angle)

for i in range(400):
    for j in range(170):
        b = np.reshape(tss[:,i,j],(4,1))
        temp = np.linalg.inv( G.transpose() @ G ) @ ( G.transpose() @ b)
        vel[0,i,j] = temp[0]
        vel[1,i,j] = temp[1]
        vel[2,i,j] = temp[2]


        
