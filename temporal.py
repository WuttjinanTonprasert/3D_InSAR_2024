#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 12 00:04:20 2023

@author: wuttinan
"""

def generate_random_temporal_baselines(d_start,d_stop):
    """
    A function intended to give non-overlapping temporal baseline between dstart and dstop
    Inputs:
        d_start | str | YYYYMMDD of when to start time series
        d_stop  | str | YYYYMMDD of when to stop time series
    Returns:
        acq_dates  | list of datetimes | acquisation dates.  
        tbaselines | list of ints | temporal baselines of short temporal baseline ifgs.  First one is 0.  
    @ modified from Syinterferopy (Gaddes et al., 2019)
    """
    from datetime import datetime, timedelta
    import numpy as np

    usual_tbaselines       = [12, 12, 12, 12, 12, 12, 12, 12, 12,
                             24, 24, 24, 24, 24, 24, 36, 48, 60, 72]   # A list of possible temporal separation
    #usual_set_baselines   = [12, 12, 12, 12, 12, 12, 12, 12, 12]   # A list of possible temporal separation
    usual_set_baselines    = [12, 12, 12, 12, 12, 12, 12, 24, 24,
                             24, 24, 24, 24, 24, 24, 36, 36, 36, 48]   # A list of possible temporal separation
    dstart = datetime.strptime(d_start,'%Y%m%d')
    dstop  = datetime.strptime(d_stop,'%Y%m%d')

    acq_dates  = []
    tbaselines = []
    tnexts     = []      

    while dstart <= dstop :
        sub_dates = [dstart]
        tbaseline = int(np.random.choice(usual_tbaselines[0:]))
        tnext = int(np.random.choice(usual_set_baselines[0:]))
        dnext = dstart + timedelta(days = tbaseline) 
        if dnext < dstop :
            sub_dates.append(dnext)
            acq_dates.append(sub_dates)
            tbaselines.append(tbaseline)
            tnexts.append(tnext)
            dstart = dstart + timedelta(days = tnext)
        else:
            break   

    return acq_dates, tbaselines , tnexts

def generate_homogeneous_random_temporal_baselines(d_start,d_stop):
    """
    A function intended to give non-overlapping temporal baseline between dstart and dstop
    Inputs:
        d_start | str | YYYYMMDD of when to start time series
        d_stop  | str | YYYYMMDD of when to stop time series
    Returns:
        acq_dates  | list of datetimes | acquisation dates.  
        tbaselines | list of ints | temporal baselines of short temporal baseline ifgs.  First one is 0.  
    @ modified from Syinterferopy (Gaddes et al., 2019)
    """
    from datetime import datetime, timedelta
    import numpy as np

    usual_tbaselines       = [12, 12, 12, 12, 12, 12, 12, 12, 12,
                             24, 24, 24, 24, 24, 24, 36, 48, 60, 72]   # A list of possible temporal separation
    dstart = datetime.strptime(d_start,'%Y%m%d')
    dstop  = datetime.strptime(d_stop,'%Y%m%d')

    acq_dates  = [dstart]
    dcurrent   = acq_dates[-1]
    tbaselines = []
    tnexts     = []      

    while dcurrent <= dstop :
        # selecting baseline first:
        tbaseline = int(np.random.choice(usual_tbaselines[0:]))
        dcurrent = dcurrent + timedelta(days = tbaseline)                           # add the temp baseline to find the new date   
        if dcurrent < dstop:                                                        # check we haven't gone past the end date 
            acq_dates.append(dcurrent)                                              # if we haven't, record            
            tbaselines.append(tbaseline)
            tnexts.append(tbaseline)
        else:
            break                                                                # remember to exit the while if we have got to the last date

    return acq_dates, tbaselines , tnexts

def generate_temporal_baselines_subset(d_start,d_stop):
    '''
    This function intends to generate set of temporal baselines, which made up of multiple independent subset
    1. each subset consist of three temporal baselines, each of which connect to one another.
    2. temporal baseline separation ranging from 12 , 18 , 24 , 36 and 72 days
    # INPUT
    dstart | datetime | date start of temporal baselines
    dstop  | datetime | date stop of temporal baselines
    # OUTPUT
    acq_dates | list of datetimes in subset | acquisation dates
    tbaselines | list of ints | temporal baselines for ifgs. 
    # History
    Sat 30 Dec 2023 | created | Wuttinan Tonprasert
    '''
    from datetime import datetime, timedelta
    import numpy as np

    usual_subset_baselines = [12, 12, 12, 12, 12, 12, 12, 12, 12, 
                              24, 24, 24, 24, 24, 24, 36, 48, 60, 72]   # A list of possible temporal separation
    usual_set_baselines    = [12, 12, 12, 12, 12, 12, 12, 12, 12,
                              24, 24, 24, 24, 24, 24, 36, 48, 48, 60]   # A list of possible temporal separation
    acq_dates = []

    dstart = datetime.strptime(d_start, '%Y%m%d')
    dstop  = datetime.strptime(d_stop, '%Y%m%d')
    dcurrent = dstart
    tbaseline_set = 0

    #
    acq_dates = []
    tba_dates = []
    tba_setdates = []
    #
    while dcurrent <= dstop :
        subset_node = 0
        subset_acq = []
        subset_tda = []
        d_next = dcurrent
        while subset_node < 3 and dcurrent <= dstop:
            #subset_acq.append(dcurrent)
            tbaseline = int(np.random.choice(usual_subset_baselines))
            d_next = d_next + timedelta(days = tbaseline)
            if d_next <= dstop :
                subset_acq.append(dcurrent)
                if subset_node < 2 :
                    subset_tda.append(tbaseline)
                else :
                    subset_tda.append(np.sum(subset_tda))
                subset_node = subset_node + 1
            else:
                break
        if len(subset_acq) == 3 and len(subset_tda) == 3 :
            '''
            regulate if-clause to identify only subset at which has member of 3 only.
            '''
            acq_dates.append(subset_acq)
            tba_dates.append(subset_tda)
            tba_setdates.append(tbaseline_set)
        tbaseline_set = int(np.random.choice(usual_set_baselines))
        dcurrent = dcurrent + timedelta(days = tbaseline_set)
    
    return acq_dates, tba_dates, tba_setdates

def temporal_baselines_subset_plot(subset_baselines,set_baselines,save_fig):

    '''
    A function intends to visualise temporal baseline generated by a function "generate_temporal_
    baselines_subset"
    # INPUT
    dstart | datetime | date start of temporal baselines
    dstop  | datetime | date stop of temporal baselines
    # OUTPUT
    ___plt.plot____ 
    # History
    Sat 30 Dec 2023 | created | Wuttinan Tonprasert
    '''
    import matplotlib.pyplot as plt
    import numpy as np
    # Image initialising
    fig = plt.figure()
    ax  = fig.add_subplot()
    point0 = 0
    for order,base in enumerate(set_baselines):
        point0 = point0 + base
        # set 1
        point1 = point0 + subset_baselines[order][0]
        ax.scatter([point0, point1],[order, order],c='red')
        ax.plot([point0, point1],[order, order],c='red')
        # set 2
        point2 = point0 + subset_baselines[order][0] + subset_baselines[order][1]
        ax.scatter([point1, point2],[order + 0.3 , order + 0.3],c='red')
        ax.plot([point1, point2],[order + 0.3 , order + 0.3],c='red')
        # set 3
        ax.scatter([point0,point2],[order + 0.6 , order + 0.6],c='red')
        ax.plot([point0,point2],[order + 0.6 , order + 0.6],c='red')
    
    ax.set_ylabel('Temporal baselines Order/Suborder')
    ax.set_xlabel('Days')
    ax.set_ylim(-0.1,15)
    ax.set_xlim(0,400)
    plt.show()

def temporal_baselines_plot(tbaseline , tnext ,save_fig):
    '''
    A function intends to visualise temporal baseline generated by a function "generate_temporal_
    baselines_subset"
    # INPUT
    dstart | datetime | date start of temporal baselines
    dstop  | datetime | date stop of temporal baselines
    # OUTPUT
    ___plt.plot____ 
    # History
    Sat 30 Dec 2023 | created | Wuttinan Tonprasert
    '''
    import matplotlib.pyplot as plt
    import numpy as np
    # Image initialising
    fig = plt.figure()
    ax  = fig.add_subplot()
    point0 = 0
    for order,base in enumerate(tbaseline):
        if order == 0 :
            point0 = 0
            point1 = point0 + base
            #print(order,'tbaselin',base,'tnext', tnext[order] ,'start',point0,'stop',point1)
            ax.scatter([point0, point1],[order, order],c='red')
            ax.plot([point0, point1],[order, order],c='red')
            point0 = point0 + tnext[order]
        else :
            point1 = point0 + base
            #print(order,'tbaselin',base,'tnext', tnext[order] ,'start',point0,'stop',point1)
            ax.scatter([point0, point1],[order, order],c='red')
            ax.plot([point0, point1],[order, order],c='red')
            point0 = point0 + tnext[order]
        # set 1
    ax.set_ylabel('Temporal baselines Order/Suborder')
    ax.set_xlabel('Days')
    '''
    ax.set_ylim(-0.1,15)
    ax.set_xlim(0,400)
    '''
    plt.show()
    
def temporal_baselines_network(dstart,dstop,directory):
    '''
    this function intend to generated a network of temporal baselines.
    # INPUT
    dstart | datetime.dtype | start dates
    dstop  | datetime.dtype | stop dates
    # OUTPUT
    a list of temporal baselines network.
    # History
    @ Sun 21 Dec 2023 | Created | Wuttinan Tonprasert 
    '''
    from datetime import datetime, timedelta
    import numpy as np
    from pathlib import Path
    # set up seed to be 10
    np.random.seed(10)

    def SAR_acq(dstart, dstop):
        '''
        Generate 'Temporal' and 'Perpendicular' baselines
        '''
        temporal_baselines = [12, 12, 12, 12, 12, 12, 12, 12] # Number of days separation
        spatial_baselines  = [-150,-150, -100, -100, -100, -50, -50, -50, -50, 0, 0, 0, 50, 50, 50, 50, 100, 100, 150, 150]

        from datetime import datetime, timedelta
        import numpy as np

        dstart = datetime.strptime(dstart,'%Y%m%d')
        dstop  = datetime.strptime(dstop,'%Y%m%d')

        temporal = []   # base separation
        coor_set  = []   # 2D list : acq_dates, perpendicular baselines  
        num = 0   

        while dstart <= dstop :
            tbaseline = int(np.random.choice(temporal_baselines[0:]))
            sbaseline = int(np.random.choice(spatial_baselines[0:]))
            #dstart    = dstart + timedelta(days = tbaseline) 
            if dstart < dstop :
                coor_set.append([dstart,sbaseline])
                temporal.append(tbaseline)
                dstart = dstart + timedelta(days = tbaseline) 
                num = num + 1
            else:
                break 

        cov_Lc = np.random.uniform(10000,300000,num)

        return coor_set, temporal, cov_Lc
    
    def SAR_network(coor_set,temporal,directory):
        '''
        The function showed a plot of SAR network.
        '''
        import matplotlib.pyplot as plt
        
        ifg_list = []
        num_pairs = 0
        fig, ax = plt.subplots()
        ax.scatter(coor_set[:,0],coor_set[:,1],c='r')
        coo = len(coor_set)
        for i,j in enumerate(coor_set):
            d_init = j[0]
            d_target = d_init + timedelta(days = 72)
            b = i + 1
            if d_target >= coor_set[-1,0]:
                d_target = coor_set[-1,0]
                if b == coo:
                    break

            while coor_set[b,0] <= d_target and i < b :
                ifg_list.append([ [coor_set[i,0], coor_set[i,1], np.sum(temporal[0:i+1]), i], [coor_set[b,0], coor_set[b,1], np.sum(temporal[0:b+1]), b]])
                #print(d_init,i,b,d_target)
                ax.plot([coor_set[i,0], coor_set[b,0]],[coor_set[i,1],coor_set[b,1]],c='r')
                b = b + 1
                num_pairs = num_pairs + 1
                if b == coo:
                    break

        ax.plot(coor_set[:,0],coor_set[:,1],c='r')
        ax.set_ylim(-300,300)
        fig.savefig( Path('./dataset') /f'SAR_network_{directory}')
        plt.close()
        return ifg_list, num_pairs
    
    coor_set, temporal, cov_Lc = SAR_acq(dstart,dstop)
    ifg_list, num_pairs = SAR_network(np.array(coor_set),np.array(temporal),directory)
    
    return ifg_list, num_pairs, cov_Lc

    
