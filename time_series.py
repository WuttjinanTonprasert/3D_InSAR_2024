#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 12 00:04:20 2023

@author: wuttinan
"""     
def generate_time_series_network(params):
    '''
    # INPUT
    params = {
        'heading'   :     heading,
        'look_angle':     look_angle,
        'choice'    :     'right_asc',
        'lon_mg'    :     X,
        'lat_mg'    :     Y,
        'vel_day'   :     V,
        'water_mask':     C,
        'ifg_network' :   ifgs,      # [[ dstart, separation, time length ],[ dstop, separation, time length ]]
        'num_pairs' :     num_pairs,
        'ph_disturb':     0.005,     # Atmospheric disturbance in metres.
        'cov_Lc'    :     200000,    # Covariance length in metres.
        'acq_dates' :     acq_dates, 
        'subset_baselines'  :    subset_baselines,
        'set_baselines'     :    set_baselines,
        'dataset_directory' :    './dataset',
        'dump_directory'    :    'right_asc'
        }
    # OUTPUT
    ___projection function___
    # History
    1 Jan 2023 | Created | Wuttinan Tonprasert
    }
    '''
    
    from inspy import projection_into_line_of_sight, generate_los
    from syinterferopy.syinterferopy import atmosphere_turb
    from alive_progress import alive_bar
    from pathlib import Path
    import numpy as np
    import pickle
    import os

    if Path(Path(params['dataset_directory']) / Path(params['dump_directory'])).exists() :
        print('the directory exist ... progress ...')
    
    else :
        os.mkdir(Path(params['dataset_directory']) / Path(params['dump_directory']))

    def plot(X,Y,C,coh,name):
        '''
        temporally plot functions.
        '''
        from mpl_toolkits.axes_grid1 import make_axes_locatable
        import matplotlib.pyplot as plt
        import matplotlib

        fig, ax = plt.subplots(figsize=(2.5,5))
        matplotlib.rc('image',cmap='viridis')
        min = np.percentile(C[:,:],10)
        max = np.percentile(C[:,:],90)
        C[:,:] = np.where(coh[:,:] == 0, np.nan, C[:,:])
        pc = ax.pcolormesh(X,Y,C,vmin=min,vmax=max)
        ax.set_xlabel('latitude [degree]')
        ax.set_ylabel('longitude [degree]')
        divider = make_axes_locatable(ax)
        cax = divider.append_axes('bottom', size='5%', pad=0.65)
        fig.colorbar(pc, cax=cax, orientation='horizontal')
        fig.savefig( Path(params['dataset_directory']) / Path(params['dump_directory']) / Path(name))
        plt.close()
        return 0
    
    def atmospheric_noise(params):
        '''
        generating atmospheric function for each of which SAR images.
        '''
        folder = 'atmospheric'

        if Path(Path(params['dataset_directory']) / Path(params['dump_directory']) / Path(folder) ).exists() :
            print('the directory exist ... progress ...')
    
        else :
            os.mkdir(Path(params['dataset_directory']) / Path(params['dump_directory']) / Path(folder))


        atmospheric = np.zeros([len(params['cov_Lc']), params['lon_mg'].shape[0], params['lon_mg'].shape[1]])
        with alive_bar(len(params['cov_Lc'])) as bar:
            for order,cov_Le in enumerate(params['cov_Lc']):       
                atmospheric[order,:,:] = atmosphere_turb(1, params['lon_mg'], params['lat_mg'], verbose=True, mean_m = params['ph_disturb'],method = 'cov', 
                                            cov_Lc = cov_Le, cov_interpolate_threshold = 10000)     # generated atmospheric disturbunce.
                plot(params['lon_mg'],params['lat_mg'],atmospheric[order,:,:],params['water_mask'], folder+'/atmosphere' + str(order))
                bar()
        
        return atmospheric

    def generate_time_series(params):
        '''
        # INPUT
        # OUTPUT
        ___projection function___
        # History
        1 Jan 2023 | Created | Wuttinan Tonprasert
        }
        '''

        defo_ts = np.zeros([params['num_pairs'], params['lon_mg'].shape[0],params['lon_mg'].shape[1]])
        print('Generating Atmospheric')
        atmos = atmospheric_noise(params)
        print('Generating Ifgs')
        with alive_bar(params['num_pairs']) as bar:
            for ifg_order,a in enumerate(params['ifg_network']):
                #print(a[:,-1])
                sar_images = np.zeros([params['lon_mg'].shape[0],params['lon_mg'].shape[1], 2])
                for network_ifg, b in enumerate(a):
                    datetime          = b[0]  # return data type 'datetime' either start or stop dates.
                    spatial_baselines = b[1]  # spatial baselines position for each position.
                    time_length       = b[2]  # time length since the start of the observation.
                    pairs             = b[3]  # Covariance length of the atmospheric.
                    # Performing projection in a SAR image.
                    los  =  generate_los(params['heading'], params['look_angle'], params['choice']) # Line-of-sight unit vector.
                    vel  =  params['vel_day'] * time_length                                         # multipy velocity per a day with number of days since observations.
                    sar_images[:,:,network_ifg]  =  projection_into_line_of_sight(vel,los) + atmos[pairs,:,:]

                bar()
                # end---
                defo_ts[ifg_order,:,:] = sar_images[:,:,1] - sar_images[:,:,0]
                plot(params['lon_mg'],params['lat_mg'],defo_ts[ifg_order,:,:],params['water_mask'],'ifgs_'+ str(ifg_order))  # Disable this sections.

        # Save deformation time series into the folder.
        with open( Path(params['dataset_directory']) / Path(params['dump_directory']) / f'defo_ts.pkl' , 'wb') as file:
                pickle.dump(defo_ts, file)

        return defo_ts
    
    def read_time_series(params):
        '''
        # Use this function if the file is already exist.
        '''
        file_name = params['dataset_directory'] + '/' + params['dump_directory'] + '/defo_ts.pkl'
                  # Deformation load from file.
        with open( Path(file_name), 'rb') as file:
            defo_ts = pickle.load(file)
        return defo_ts
    
    try:
         return read_time_series(params)
    except:
         return generate_time_series(params)

def acquire_time_series(params):
    '''
    # INPUT
    params = {
        'heading'   :     heading,
        'look_angle':     look_angle,
        'choice'    :     'right_asc',
        'lon_mg'    :     X,
        'lat_mg'    :     Y,
        'vel_day'   :     V,
        'water_mask':     C,
        'ph_disturb':     0.005,     # Atmospheric disturbance in metres.
        'cov_Lc'    :     200000,    # Covariance length in metres.
        'acq_dates' :     acq_dates, 
        'subset_baselines'  :    subset_baselines,
        'set_baselines'     :    set_baselines,
        'dataset_directory' :    './dataset',
        'dump_directory'    :    'right_asc'
        }
    # OUTPUT
    ___projection function___
    # History
    1 Jan 2023 | Created | Wuttinan Tonprasert
    }
    '''
    
    from inspy import projection_into_line_of_sight, generate_los
    from syinterferopy.syinterferopy import atmosphere_turb
    from pathlib import Path
    import numpy as np
    import pickle
    import os

    if Path(Path(params['dataset_directory']) / Path(params['dump_directory'])).exists() :
        print('the directory exist ... progress ...')
    
    else :
        os.mkdir(Path(params['dataset_directory'] / params['dump_directory']))

    def generate_time_series(params):
        '''
        # INPUT
        # OUTPUT
        ___projection function___
        # History
        1 Jan 2023 | Created | Wuttinan Tonprasert
        }
        '''

        defo_ts = np.zeros([len(params['acq_dates']), params['lon_mg'].shape[0],params['lon_mg'].shape[1],3])
        order_num = 0
        for order,baselines in enumerate(params['subset_baselines']):
            for suborder,subbaselines in enumerate(baselines):
                # Generate Line-of-sight unit vector.
                los  =  generate_los(params['heading'][order_num], params['look_angle'][order_num], params['choice'])
                # Coverted velocity fields per day into displacement over temporal baselines separations
                vel  =  params['vel_day'] * subbaselines
                # Projected into line-of-sight
                projected_array = projection_into_line_of_sight(vel,los)
                # generated atmospheric disturbunce.
                atmos = atmosphere_turb(1 , params['lon_mg'], params['lat_mg'], mean_m = params['ph_disturb'],
                                    cov_interpolate_threshold = 1e4, cov_Lc = params['cov_Lc'])
                # add atmospheric noise into the dataset
                projected_array = projected_array + atmos
                # Incremental part
                order_num = order_num + 1
                # save projected Array into defo_ts
                defo_ts[order,:,:,suborder] = projected_array
                # save file:
                with open( Path(params['dataset_directory']) / Path(params['dump_directory']) / f'defo_{order}_{suborder}.pkl' , 'wb') as file:
                    pickle.dump(projected_array, file)
                print('progress___',order,'/',suborder,'___out_of___',len(params['acq_dates']),'/',order_num)

        with open( Path(params['dataset_directory']) / Path(params['dump_directory']) / f'heading_looking.pkl', 'wb') as file:

                pickle.dump(params['heading'], file)
                pickle.dump(params['look_angle'], file)   
        # Return deformation time series.
        return defo_ts
    
    def read_time_series(params):
        '''
        # Use this function if the file is already exist.
        '''
        defo_ts = np.zeros([len(params['acq_dates']), params['lon_mg'].shape[0],params['lon_mg'].shape[1],3])
        for order in range(len(params['acq_dates'])):
             for suborder in range(3):
                  file_name = params['dataset_directory'] + '/' + params['dump_directory'] + '/defo_' + str(order) + '_' + str(suborder) + '.pkl'
                  # Deformation load from file.
                  with open( Path(file_name), 'rb') as file:
                       defo_ts[order,:,:,suborder] = pickle.load(file)
        return defo_ts
    
    try:
         return read_time_series(params)
    except:
         return generate_time_series(params)

def acquire_set_time_series(params):
    '''
    # INPUT
    params = {
        'heading'   :     heading,
        'look_angle':     look_angle,
        'choice'    :     'right_asc',
        'lon_mg'    :     X,
        'lat_mg'    :     Y,
        'vel_day'   :     V,
        'water_mask':     C,
        'ph_disturb':     0.005,     # Atmospheric disturbance in metres.
        'cov_Lc'    :     200000,    # Covariance length in metres.
        'acq_dates' :     acq_dates, 
        'subset_baselines'  :    subset_baselines,
        'set_baselines'     :    set_baselines,
        'dataset_directory' :    './dataset',
        'dump_directory'    :    'right_asc'
        }
    # OUTPUT
    ___projection function___
    # History
    1 Jan 2023 | Created | Wuttinan Tonprasert
    }
    '''
    
    from inspy import projection_into_line_of_sight, generate_los
    from syinterferopy.syinterferopy import atmosphere_turb
    import matplotlib.pyplot as plt
    from pathlib import Path
    import numpy as np
    import pickle
    import os

    if Path(Path(params['dataset_directory']) / Path(params['dump_directory'])).exists() :
        print('[1] | Checked | The directory exist >> read files')
    
    else :
        print('[1] | Unchecked | The directory didnot exist >> directory created')
        os.mkdir(Path(params['dataset_directory'] / params['dump_directory']))

    def generate_set_time_series(params):
        '''
        # INPUT
        # OUTPUT
        ___projection function___
        # History
        1 Jan 2023 | Created | Wuttinan Tonprasert
        }
        '''

        defo_ts = np.zeros([len(params['acq_dates']), params['lon_mg'].shape[0],params['lon_mg'].shape[1]])
        order_num = 0
        for order in range(len(params['acq_dates'])):
            # Generate Line-of-sight unit vector.
            los  =  generate_los(params['heading'][order_num], params['look_angle'][order_num], params['choice'])
            # Coverted velocity fields per day into displacement over temporal baselines separations
            vel  =  params['vel_day'] * params['subset_baselines'][order]
            # Projected into line-of-sight
            projected_array = projection_into_line_of_sight(vel,los)
            # generated atmospheric disturbunce.
            atmos = atmosphere_turb(1 , params['lon_mg'], params['lat_mg'], method='cov', mean_m = params['ph_disturb'],
                                water_mask = params['water_mask'], cov_interpolate_threshold = 1e4, cov_Lc = params['cov_Lc'])
            # add atmospheric noise into the dataset
            projected_array = projected_array + atmos
            # Incremental part
            order_num = order_num + 1
            # save projected Array into defo_ts
            defo_ts[order ,: ,: ] = projected_array
            # save file:
            with open( Path(params['dataset_directory']) / Path(params['dump_directory']) / f'set_defo_{order}.pkl' , 'wb') as file:
                pickle.dump(projected_array, file)
            print('progress___',order,'/','___out_of___',params['subset_baselines'][order],'/',order_num)

        with open( Path(params['dataset_directory']) / Path(params['dump_directory']) / f'set_heading_looking.pkl', 'wb') as file:

                pickle.dump(params['heading'], file)
                pickle.dump(params['look_angle'], file)   
        # Return deformation time series.
        return defo_ts
    
    def read_set_time_series(params):
        '''
        # Use this function if the file is already exist.
        '''
        defo_ts = np.zeros([len(params['acq_dates']), params['lon_mg'].shape[0],params['lon_mg'].shape[1]])
        for order in range(len(params['set_baselines'])):
                file_name = params['dataset_directory'] + '/' + params['dump_directory'] + '/set_defo_' + str(order) + '.pkl'
                # Deformation load from file.
                with open( Path(file_name), 'rb') as file:
                    defo_ts[order ,: ,: ] = pickle.load(file)
        return defo_ts
    
    try:
         print('[2] | Checked | Read projected velocity from a dump directory')
         return read_set_time_series(params)
    except:
         print('[2] | Checked | Projected velocity didnot exists >> generate new velocity files')
         return generate_set_time_series(params)

def small_baseline_subset_inversion(params, tss):
     '''
     This function, firstly, performs kernel matrix function.
     '''
     from alive_progress import alive_bar
     import matplotlib.pyplot as plt
     import numpy as np
     from pathlib import Path
     import pickle 
    
     print('[3] | Checked | Initialised Small baseline subset Inversion')

     def kernel_matrix(params):
        '''
        This function return a kernel matrix for Small Baselin Subset Inversion (SBAS)
        '''
        from datetime import datetime, timedelta

        dstart = params['ifg_network'][0,0,0]
        dstop = params['ifg_network'][-1,1,0]
        ncols  = int((dstop - dstart).days / 12)                 # numbers of Kernel matrix column, 12 is the minimal interval
        nrows  = params['num_pairs']                             # numbers of Kernal matrix row.
        Kernel_matrix = np.zeros([nrows, ncols])                 # initiaslised matrix with zeros members.
        
        for ifg_id, info in enumerate(params['ifg_network']):
            pair_start = info[0,0]
            pair_stop = info[1,0]
            start_position = int((pair_start - dstart).days /12)
            stop_position  = int((pair_stop - dstart).days /12)
            Kernel_matrix[ifg_id, start_position:stop_position] = 12    # number of days

        return Kernel_matrix

     def kernel_matrix_old(acq_dates, subset_baselines, set_baselines):
        '''
        This function return a kernel matrix for Small Baselin Subset Inversion (SBAS)
        '''
        from datetime import datetime, timedelta

        dstart = acq_dates[0][0]
        dstop  = acq_dates[-1][-1]
        ncols  = int((dstop - dstart).days / 12)                 # numbers of Kernel matrix column, 12 is the minimal interval
        nrows  = len(subset_baselines)                           # numbers of Kernal matrix row.
        Kernel_matrix = np.zeros([nrows, ncols])                 # initiaslised matrix with zeros members.
        init_point = 0
        for order in range(len(acq_dates)):
            if order == 0 :
                dcurrent   = init_point
                stop_point = init_point + subset_baselines[order]
                #print(order,'tbaselin',subset_baselines[order],'tnext', set_baselines[order] ,'start', init_point,'stop', stop_point)
                init_point = init_point + set_baselines[order]
                
            else :
                stop_point = init_point + subset_baselines[order]
                dcurrent   = init_point
                #print(order,'tbaselin',subset_baselines[order],'tnext', set_baselines[order] ,'start', init_point,'stop', stop_point)
                init_point = init_point + set_baselines[order]
            
            while dcurrent < stop_point:
                 Kernel_matrix[order, int(dcurrent/12)] = 12          # set the Kernel matrix within a range of temporal baselines to be 12 days
                 #print('dcurrent', dcurrent, dcurrent/12,'dstop',stop_point)
                 dcurrent = dcurrent + 12
        return Kernel_matrix
     
     def data_matrix(tss,xcoor,ycoor):
          '''
          This function construct a whole data matrix at xcoor and ycoor
          '''
          import numpy as np
          ndim, nx, ny = tss.shape
          return tss[: ,xcoor ,ycoor].reshape(ndim,1)
     
     def least_square_inversion(G,d):
          '''
          Due to an overlapping between these two dataset, typically lead to Singular Matrix.
          '''
          import numpy as np
          #print( G.transpose().shape, d.shape )
          m = np.linalg.inv( G.transpose() @ G ) @ ( G.transpose() @ d )
          return m
     
     def tikhonov_least_sqaure_inversion(G,d,u):
        '''
        This function performs tikhonov regularisation, by adding solution smoothness.
        as the G is rank-deficient matrix.
        '''
        import numpy as np
        wa,wb = G.shape
        GTG = np.linalg.inv( G.transpose() @ G + u * np.identity(wb))
        m_est = GTG @ G.transpose() @ d
        return m_est
     
     def tikhonov_svd(G ,U , eigenS, V, d):
        return 0
     
     def singular_value_decomposition(G):
        '''
        This function returns singular decomposition value (SVD)
        '''
        U, eigenS, Vt = np.linalg.svd(G)
        V = Vt.transpose()
        '''
        print('[5]____________[SVD initialised]___________\n')
        print('     Matrix G (M x N): M :', G.shape[0],' nrows')
        print('                     : N :', G.shape[1],' ncols')
        print('      SVD result     : U :', U.shape)
        print('                     : Vt:', Vt.shape)
        print('     Member of EigenValue: ', len(eigenS), '(if singular matrix found agains, this part?)')
        print('     There are Model Null space', G.shape[1] - G.shape[0])
        print('Exception : function return[1]\n')
        '''
        return U, eigenS, V
     
     def moore_penrose_inversion(G ,U , eigenS, V, d):
        '''
        SVD (G) = [ UP  UO ][ SP  0 ][ VP  VO ]^T
                            [  0  0 ]
        
          G    =   UP x SP x VPT
          G-g  =   VP x SP-1 x UPT
        '''
        eigenV = len(eigenS)
        Up     = U[:,0:eigenV]
        Vp     = V[:,0:eigenV]
        Sp     = np.diag(eigenS[0:eigenV])

        matrix_norm = np.linalg.norm( G - Up @ Sp @ Vp.transpose())
        G_invg = Vp @ np.linalg.inv(Sp) @ Up.transpose()
        m_hat  = G_invg @ d
        return m_hat
     
     #G = kernel_matrix(params['acq_dates'], params['subset_baselines'], params['set_baselines'])    # Retrieve the Kernel matrix
     G = kernel_matrix(params)
     
     fig, ax = plt.subplots(figsize=(5,10),ncols=1, nrows=2)
     ax[0].imshow(G)   # Plot a Resolution matrix.
     #ax[1].imshow( G.transpose() @ G ) 
     ax[1].imshow( np.linalg.inv(G.transpose() @ G) @ ( G.transpose() @ G ) ) 
     fig.savefig( Path(params['dataset_directory'])/ Path(params['dump_directory']) / f'Matrix_G' )
     plt.close()
    
     sbas_matrix = np.zeros([ G.shape[1], params['lon_mg'].shape[0], params['lon_mg'].shape[1]]) # initiate SBAS matrix.
     u = 0.01

     try:
         with open( Path(params['dataset_directory']) / Path(params['dump_directory']) / f'SBAS_time_series.pkl' , 'rb') as file:
             sbas_matrix = pickle.load(file)
         return sbas_matrix
     
     except:
        with alive_bar(params['lon_mg'].shape[0] * params['lat_mg'].shape[1]) as bar:
            for nlons in range(params['lon_mg'].shape[0]):
                for nlats in range(params['lat_mg'].shape[1]):
                    if params['water_mask'][nlons,nlats] == 1 :
                        d = data_matrix(tss ,nlons ,nlats)
                        # The SBAS velocity fields is tends to diffuse, try to use Tikhonov regulations.
                        try:
                            #sbas_matrix[:,nlons,nlats] = least_square_inversion(G,d)
                            sbas_matrix[:,nlons,nlats] = tikhonov_least_sqaure_inversion(G,d,params['tikhonov_ratio'])
                            # Performing SBAS with priority from least square inversion to singular value decomposition
                        
                        except:   
                            U, eigenV, V = singular_value_decomposition(G)
                            m = moore_penrose_inversion(G ,U ,eigenV ,V ,d)
                            for ordi, xcv in enumerate(m):
                                sbas_matrix[ordi, nlons, nlats] = xcv
                    bar()
                    
            # This typically be enabled, but disabled due to iterate tikhonov_ratio.
            '''
            with open( Path(params['dataset_directory']) / Path(params['dump_directory']) / f'SBAS_time_series.pkl' , 'wb') as file:
                pickle.dump(sbas_matrix, file)
            '''
     
        return sbas_matrix


                
                                                                      
     