#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 11 23:41:52 2023

@author: wuttinan
"""
def read_file(file_name):
    '''
    Parameters
    ----------
    file_name : Char64
        file name that contained velocity fields.

    Returns
    -------
    @Wuttinan Tonprasert, Mon 11th Dec 2023
    '''
    import numpy as np
    file = np.loadtxt(file_name,delimiter='\t')
     
    # Extract range of unique set of latitude and longitude
    lat = np.unique(file[:,0])
    lon = np.unique(file[:,1])
    velocities_field = np.zeros([len(lon),len(lat),3])
    coherent_mask    = np.zeros([len(lon),len(lat)])
    # converge the whole dataset into format.
    i = 0
    for column in range(len(lat)):
        for row in range(len(lon)):
            if file[i,0] == lat[column] and file[i,1] == lon[row] :
                '''
                Condition: if the file is the same latitude and longitude.
                '''
                velocities_field[row,column,0] = file[i,2]
                velocities_field[row,column,1] = file[i,3]
                velocities_field[row,column,2] = file[i,4]
                coherent_mask[row,column] = 1
                if i <= 690345 :
                     '''
                     specific to this dataset to aviod the error.
                     '''
                     i = i + 1
            
    X, Y = np.meshgrid(lat,lon)
    # Save file
    np.save('latitude.npy',X)
    np.save('longitude.npy',Y)
    np.save('Velocity_fields.npy',velocities_field)
    np.save('Coherent.npy',coherent_mask)

def look_angle(size):
    '''

    Returns
    -------
    returned Random uniform look angle between 25 to 37 degree.

    '''
    import numpy as np
    return np.random.uniform(25,37,size)

def a_look_angle(lats_sat,lons_sat,lats_sta,lons_sta,alt):
    # a function to return look angle (a) : 
    r = 700 # kilometres of earth radii.
    # this formulated r = r1 * np.sqrt((x1-x2)**2 + (y1-y2)**2 + (z1-z2)**2)
    import numpy as np
    lons_sat = np.deg2rad(lons_sat)
    lats_sat = np.deg2rad(lats_sat)
    lons_sta = np.deg2rad(lons_sta)
    lats_sta = np.deg2rad(lats_sta)
    
    xsat = np.cos(lons_sat)*np.cos(lats_sat)
    ysat = np.sin(lons_sat)*np.cos(lats_sat)
    zsat = np.sin(lats_sat)
    
    xpos = np.cos(lons_sta)*np.cos(lats_sta)
    ypos = np.sin(lons_sta)*np.cos(lats_sta)
    zpos = np.sin(lats_sta)
    distance = r * np.sqrt((xpos - xsat)**2 + (ypos - ysat)**2 + (zpos - zsat)**2)
    return np.arctan(distance/alt)

def a_random_satellite(lats_seeds,lons_seeds):
    import numpy as np
    rng = np.random.default_rng(seed=None)
    a = 3*rng.random()
    b = 5.5*rng.random()
    theta = 2*np.pi * rng.random()
    
    x_seeds = lats_seeds + a*np.cos(theta) + b*np.sin(theta)
    y_seeds = lons_seeds - a*np.sin(theta) + b*np.sin(theta)
    return x_seeds, y_seeds

def generate_los(heading, look_angle, choice):
    '''
    Parameters
    ----------
    heading : float64
        satellite heading angle
    look_angle : float64
        look_angle took earlier
    choice : either 'asc' or 'des'

    Returns
    -------
    R_vec : TYPE

    '''
    import numpy as np
    if choice == 'right_asc':
        los = np.array([np.sin(-heading)*np.sin(look_angle),
                          np.cos(-heading)*np.sin(look_angle),
                          -np.cos(look_angle)])
    elif choice == 'right_des':
        los = np.array([np.sin(-heading)*np.sin(look_angle),
                          np.cos(-heading)*np.sin(look_angle),
                          -np.cos(look_angle)])
    elif choice == 'left_asc':
        los = np.array([np.sin(np.pi - heading)*np.sin(look_angle),
                          np.cos(np.pi - heading)*np.sin(look_angle),
                          -np.cos(look_angle)])
    elif choice == 'left_des':
        los = np.array([np.sin(np.pi - heading)*np.sin(look_angle),
                          np.cos(np.pi - heading)*np.sin(look_angle),
                          -np.cos(look_angle)])
        
    return los
def projection_into_line_of_sight(x,los):
    '''
    This function into to projected values in array 'x' into the line-of-sight unit vector
    # Input
    x | 3D array | velocity fields array.
    # Return
    # History
    31 Dec 2023 | created | Wuttinan Tonprasert
    '''
    import numpy as np
    lenx, leny, lenz = x.shape
    projected_array = np.zeros([lenx, leny])
    for row in range(lenx):
        for col in range(leny):
            projected_array[row,col] = x[row,col,0] * los[0] + x[row, col, 1] * los[1] + x[row, col, 2] * los[2]
    return projected_array








    