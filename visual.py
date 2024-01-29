#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 12 00:04:20 2023

@author: wuttinan
"""
def visual_time_series(xcoor,ycoor,tss,coh,params):
    '''
    This function intend to visualise the all generated time series.
    # INPUT
    xcoor | 2D float64 array | X-coordinate of the velocity field
    ycoor | 2D float64 array | Y-coordinate of the velocity field
    tss   | 3D float64 array | temporal velocity fields 
    # OUTPUT
    ___matplotlib.pyplot____
    # History
    28 Jan 2024 | created | Wuttinan Tonprasert
    '''
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    import matplotlib.animation as animation
    import matplotlib.pyplot as plt
    from pathlib import Path
    import numpy as np
    import matplotlib
    import os

    def gif_creation(tss,params,norder):
        '''
        # Gif creation function. intending to show temporal variation in tss array.
        # INPUT
            # tss | 3D array | time series to visualise
            # params | dictionary | related parameters
            # norder | uint32     | time frame for a time series.
        # OUTPUT
            # ____gif_named "sbas.gif"__
        # History
            # Sun 28 DEC 2024 | Created | Wuttinan Tonprasert
        '''
        min = np.percentile(tss,5)
        max = np.percentile(tss,95)
        fig, ax = plt.subplots(figsize=(2.5,5))
        pc = ax.pcolormesh(params['lon_mg'],params['lat_mg'],tss[0,:,:],vmin = min, vmax = max)
        ax.set_xlabel('latitude (degree)')
        ax.set_ylabel('longitude (degree)')
        divider = make_axes_locatable(ax)
        cax = divider.append_axes('bottom', size='5%', pad=0.65)
        bar = fig.colorbar(pc, cax=cax, orientation='horizontal')
        bar.set_label('milimetre per a day (mm/day)')

        def animate(i):
            pc.set_array(tss[i, :, :].flatten())

        anim   = animation.FuncAnimation(fig, animate, interval=500, frames= norder - 1)   # Animate the plot.
        writer = animation.PillowWriter(fps=15,metadata=dict(artist='Me'),bitrate=1800)
        anim.save( Path(params['dataset_directory']) / Path(params['dump_directory']) / f'sbas_pic' / f'sbas.gif',writer=writer)
        plt.show()
        return 0

    # Set up the directory
    if Path( Path(params['dataset_directory']) / Path(params['dump_directory']) / f'sbas_pic').exists() :
        print(' _sbas_pic_ | [folder] | status existed ')
    
    else :
        os.mkdir(Path( Path(params['dataset_directory']) / Path(params['dump_directory']) / f'sbas_pic'))

    norder, nrow, ncol= tss.shape    # Retrieve dimensional size of the function.
    fig, ax = plt.subplots(figsize=(2.5,5))     # initialise the matplotlib function.
    matplotlib.rc('image',cmap='RdBu')
    min = np.percentile(tss,5)      # Calculate the min value for a colorbar at  5 percentile
    max = np.percentile(tss,95)     # Calculate the max value for a colorbar at 95 percentile

    for order in range(norder):
        '''
        # for loop through each temporal timeframe to plot
        '''
        tss[order,:,:] = np.where(coh[:,:] == 0, np.nan, tss[order,:,:])
        pc = ax.pcolormesh(xcoor,ycoor,tss[order,:,:],vmin = min, vmax = max)
        ax.set_title(f'Time series, days {12*order} to {12*(order + 1)} days')  # axis labelling
        ax.set_xlabel('latitude (degree)')
        ax.set_ylabel('longitude (degree)')
        divider = make_axes_locatable(ax)                                       # Colour bar setting
        cax = divider.append_axes('bottom', size='5%', pad=0.65)
        bar = fig.colorbar(pc, cax=cax, orientation='horizontal')
        bar.set_label('metre per a day (m/day)')
        fig.savefig( Path(params['dataset_directory']) / Path(params['dump_directory']) / f'sbas_pic' / f'sbas_savefig_{order}')
        plt.close() 
    
    gif_creation(tss,params,norder)

def los_displacement(params,sbas,name):
    '''
    # This function constructed accumulative displacement, los velocity fields from Small Baselines Subset (SBAS) inversion.
    # as vlos = v0 + vlos * t, as each of SBAS time frame, the time separation is 12 days.
    # INPUT
        # sbas   | 3D array   | time series to visualise
        # params | dictionary | related parameters
        # name   | char64     | a name for saving los_displacement
    # OUTPUT
        # ____gif_for_los_displacement____
    # History
        # Sun 28 DEC 2024 | Created | Wuttinan Tonprasert
    '''
    from inspy import projection_into_line_of_sight, generate_los
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    import matplotlib.pyplot as plt
    import matplotlib.animation as animation
    import os
    import numpy as np
    from pathlib import Path
    import matplotlib

    # Dependency functions
    def disp(days,sbas):
        '''
        the number of days should starting from days = 0 to days = norder
        '''
        return np.sum(sbas[0:days],axis=0) * 12
    
    def rms_los(params,sbas,name):
        '''
        # This function returned a displacement time series, constructing from SBAS inversion,
        # and residual between calculated and theoretical velocity fields.
        # INPUT
            # sbas   | 3D array   | time series to visualise
            # params | dictionary | related parameters
            # name   | char64     | a name for saving rms_residual
        # OUTPUT
            # ____gif_for_los_displacement____
        # History
            # Sun 28 DEC 2024 | Created | Wuttinan Tonprasert
        '''
        los  =  generate_los(params['heading'], params['look_angle'], params['choice']) # Line-of-sight unit vector.
        vel  =  projection_into_line_of_sight(params['vel_day'],los)
        norder, nrow, ncol = sbas.shape
        disp_tss = []       # accumulative displacement time series.
        rms  =  []          # uint8 - frebenius norm for calculated and theoretical accumulative displacement
        residual_tss = []   # residual - a residual map between calculated and theoretical accumulative displacement
        for days in range(norder):
            disps = disp(days, sbas)
            disp_tss.append(disps)
            residual = disps - 12 * days * vel 
            residual_tss.append(residual)
            residual = np.where( params['water_mask'][:,:] == 0, 0, residual[:,:])
            rms.append(np.linalg.norm(residual))
        
        gif_creation(params,np.array(residual_tss),norder,name + 'residual','residual between calculated and theoretical')
        
        x = np.arange(0,norder,1)
        fig,ax = plt.subplots()
        ax.scatter(x, rms,c='r',label='rms')
        ax.set_ylabel('RMS misfit (metres)')
        ax.set_xlabel('dates')
        fig.savefig( Path(params['dataset_directory']) / Path(params['dump_directory']) / f'sbas_pic' / f'{name}_rms_by_timeframe')
        plt.close()

        return np.array(disp_tss),np.array(rms)
    
    def gif_creation(params,tss,norder,name,title):
        '''
        # This function returned a gif image of tss,
        # INPUT
            # sbas   | 3D array   | time series to visualise
            # params | dictionary | related parameters
            # name   | char64     | a name for saving rms_residual
        # OUTPUT
            # ____gif_for_los_displacement____
        # History
            # Sun 28 DEC 2024 | Created | Wuttinan Tonprasert
        '''
        min = np.percentile(tss,5)
        max = np.percentile(tss,95)
        fig, ax = plt.subplots(figsize=(4.8,8))
        matplotlib.rc('image',cmap='RdBu')
        tss = np.where(params['water_mask'][:,:] == 0, np.nan, tss[:,:])
        pc = ax.pcolormesh(params['lon_mg'],params['lat_mg'],tss[0,:,:],vmin = min, vmax = max)
        ax.set_title(str(title))
        ax.set_xlabel('latitude (degree)')
        ax.set_ylabel('longitude (degree)')
        divider = make_axes_locatable(ax)
        cax = divider.append_axes('bottom', size='5%', pad=0.65)
        bar = fig.colorbar(pc, cax=cax, orientation='horizontal')
        bar.set_label('metres/day (mm/day)')

        def animate(i):
            pc.set_array(tss[i, :, :].flatten())

        anim = animation.FuncAnimation(fig, animate, interval=100, frames= norder)
        writer = animation.PillowWriter(fps=15,metadata=dict(artist='Me'),bitrate=1800)
        anim.save( Path(params['dataset_directory']) / Path(params['dump_directory']) / f'sbas_pic' / f'{name}.gif',writer=writer)
        plt.close()

        return 0
    
    def gif_validator(params,sbas,min,max,name,title):
        '''
        # This function returned a gif image of tss,
        # INPUT
            # sbas   | 3D array   | time series to visualise
            # params | dictionary | related parameters
            # name   | char64     | a name for saving rms_residual
        # OUTPUT
            # ____gif_for_los_displacement____
        # History
            # Sun 28 DEC 2024 | Created | Wuttinan Tonprasert
        '''
        norder, nrow, ncol = sbas.shape
        matplotlib.rc('image',cmap='RdBu')
        fig, ax = plt.subplots(figsize=(4.8,8))
        dispa = disp(0,sbas)
        dispa = np.where(params['water_mask'][:,:] == 0, np.nan, dispa[:,:])
        pc = ax.pcolormesh(params['lon_mg'],params['lat_mg'],dispa,vmin = min, vmax = max)
        ax.set_title(str(title))
        ax.set_xlabel('latitude (degree)')
        ax.set_ylabel('longitude (degree)')
        divider = make_axes_locatable(ax)
        cax = divider.append_axes('bottom', size='5%', pad=0.65)
        bar = fig.colorbar(pc, cax=cax, orientation='horizontal')
        bar.set_label('metres')

        def animate(i):
            disps = disp(i,sbas)
            disps = np.where(params['water_mask'][:,:] == 0, np.nan, disps[:,:])
            pc.set_array(disps.flatten())

        anim = animation.FuncAnimation(fig, animate, interval=100, frames= norder)
        writer = animation.PillowWriter(fps=15,metadata=dict(artist='Me'),bitrate=1800)
        anim.save( Path(params['dataset_directory']) / Path(params['dump_directory']) / f'sbas_pic' / f'{name}.gif',writer=writer)
        plt.close()
        return 0
    
    # Checking the directory existence.
    if Path( Path(params['dataset_directory']) / Path(params['dump_directory']) / f'sbas_pic').exists() :
        print(' progress ...')
    
    else :
        os.mkdir(Path( Path(params['dataset_directory']) / Path(params['dump_directory']) / f'sbas_pic'))

    disps_tss, rms_los = rms_los(params,sbas,name)
    gif_validator(params,sbas,-22,22,name + 'displacement','calculated accumulative displacement')

    return disps_tss, rms_los


def const_displacement(params,norder,vel,name):
    '''
    This function constructed accumulative displacement, los velocity fields from Small Baselines Subset (SBAS) inversion.
    as vlos = v0 + vlos * t
    as each of SBAS time frame, the time separation is 12 days.
    '''
    from inspy import projection_into_line_of_sight, generate_los
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    import matplotlib.pyplot as plt
    import matplotlib.animation as animation
    import numpy as np
    from pathlib import Path
    import matplotlib
    import os

    # Function
    def rms_los(params,norde,velo,name):
        '''
        this function calculates and visualises RMS error of each displacement to the dataset.
        '''
        los  =  generate_los(params['heading'], params['look_angle'], params['choice']) # Line-of-sight unit vector.
        vel_ref  =  projection_into_line_of_sight(params['vel_day'],los)
        norder = norde
        disp_tss = []
        rms  =  []
        residual_tss = []
        for days in range(norder):
            residual = (velo - vel_ref) * 12 * days 
            residual_tss.append(residual)
            residual = np.where( params['water_mask'][:,:] == 0, 0, residual[:,:])
            rms.append(np.linalg.norm(residual))
        
        gif_valid(params,np.array(residual_tss),norder,name + 'residual')
        
        x = np.arange(0,norder,1)
        fig,ax = plt.subplots()
        ax.scatter(x, rms)
        ax.set_ylabel('RMS misfit (metres)')
        ax.set_xlabel('dates')
        fig.savefig( Path(params['dataset_directory']) / Path(params['dump_directory']) / f'sbas_pic' / f'{name}_rms_by_timeframe')
        plt.close()

        return np.array(rms)
    
    def gif_valid(params,tss,norder,name):
        '''
        gif validator function.
        '''
        min = np.percentile(tss,5)
        max = np.percentile(tss,95)
        fig, ax = plt.subplots(figsize=(4.8,8))
        tss = np.where(params['water_mask'][:,:] == 0, np.nan, tss[:,:])
        pc = ax.pcolormesh(params['lon_mg'],params['lat_mg'],tss[0,:,:],vmin = min, vmax = max)
        ax.set_xlabel('latitude (degree)')
        ax.set_ylabel('longitude (degree)')
        divider = make_axes_locatable(ax)
        cax = divider.append_axes('bottom', size='5%', pad=0.65)
        bar = fig.colorbar(pc, cax=cax, orientation='horizontal')
        bar.set_label('metres')

        def animate(i):
            pc.set_array(tss[i, :, :].flatten())

        anim = animation.FuncAnimation(fig, animate, interval=100, frames= norder)
        writer = animation.PillowWriter(fps=15,metadata=dict(artist='Me'),bitrate=1800)
        anim.save( Path(params['dataset_directory']) / Path(params['dump_directory']) / f'sbas_pic' / f'{name}.gif',writer=writer)
        plt.close()
        #plt.show()

        return 0
    
     # Checking the directory existence.
    if Path( Path(params['dataset_directory']) / Path(params['dump_directory']) / f'sbas_pic').exists() :
        print(' progress ...')
    
    else :
        os.mkdir(Path( Path(params['dataset_directory']) / Path(params['dump_directory']) / f'sbas_pic'))

    rms_los = rms_los(params,norder,vel,name)
    #gif_valid(params,vel,-22,22,name + 'displacement')

    return rms_los

    
    
    




    

