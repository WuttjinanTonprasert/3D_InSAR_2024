#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 12 00:04:20 2023

@author: wuttinan
"""

import inspy as ins
import numpy as np
from pathlib import Path
import os
import pickle

data_path    = Path('./dataset')
dataset_name = Path('Venu.LiCSBASv2.mod.xyz')
ts_path = ['./dataset/right_asc', './dataset/right_des',
            './dataset/left_asc', './dataset/left_des']       # Directory to store timeseries dump.

for i in ts_path:
    if not Path(i).exists() :
        # Create directories in the list, if there no exists.
        os.mkdir(i)

if Path(data_path).exists() :
    print('File exists, proceeded')
    # Load the file data into the program
    try:
        # Try to load latitude and logitude data
        longitude = np.transpose(np.load('latitude.npy'))
        latitude  = np.transpose(np.load('longitude.npy'))
        coherent = np.transpose(np.load('Coherent.npy'))
        Velocity = np.transpose(np.load('Velocity_fields.npy'),(1,0,2))
        vel_day = Velocity/365

    except:
        # If there is no data is available
        file_link = Path(data_path / dataset_name)
        ins.read_file(file_link)
else :
    print('File does not exist')



# export area of interest : it locate at array location 

try:
    with open(data_path / f'X.pkl','wb') as lat:
        pickle.dump(latitude[700:1100,580:750],lat)

    with open(data_path / f'Y.pkl','wb') as lon:
        pickle.dump(longitude[700:1100,580:750],lon)

    with open(data_path / f'Vel_day.pkl','wb') as vel:
        pickle.dump(vel_day[700:1100,580:750,:],vel)

    with open(data_path / f'Coherent.pkl','wb') as coh:
        pickle.dump(coherent[700:1100,580:750],coh)

except:
    print('[Pre-processing] | status | Unable to save')




