#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 11 23:41:52 2023

@author: wuttinan
"""

def InSAR_extraction(sat_list, heading_angle):
    '''
    this function perform linearly inversion on the dataset.
    '''
    from inspy import generate_los
    import numpy as np

    kernel_matrix = []

    for order,directory in enumerate(sat_list):
        temp = generate_los(np.deg2rad(heading_angle[order]), np.deg2rad(34), directory)
        kernel_matrix.append(temp)

    kernel_matrix = np.array(kernel_matrix)
        
    return kernel_matrix

def displacement_least_square(disps_tss):
    '''
    This function performs least square inversion on the displacement fields.
    '''
    import numpy as np
    from alive_progress import alive_bar

    norder, nrow, ncol = disps_tss.shape
    m_est = np.zeros([2,nrow,ncol]) # estimate linear displacement [0]: average velocity, [1]: initial_velocity.
    G = np.ones([norder,2])
    for i in range(norder):
        G[i,0] = 12 * i

    GTG = np.linalg.inv(G.transpose() @ G)
    with alive_bar(nrow*ncol) as bar:
        for row in range(nrow):
            for col in range(ncol):
                d = disps_tss[:,row,col].reshape(norder,1)
                m = GTG @ ( G.transpose() @ d )
                m_est[0,row,col] = m[0]
                m_est[1,row,col] = m[1]
                bar()
    
    return m_est
