
def get_MRR_melting_layer(MRRncfile, event_start, event_stop, path_dir_out, 
                           alt_station, chi=0.7, r=1.5, sigma=20, limit_interpol=90):
    """
    
    Parameters
    ----------
    MRRncfile : netcdf
        MRR netcdf file.
    event_start : string
        Starting time [UTC] of the event in format %Y-%m-%d %H:%M:%S.
    event_stop : string
        Ending time [UTC] of the event in format %Y-%m-%d %H:%M:%S.
    path_dir_out : string
        Path of the directory to save the plots.
    alt_station : float
        Altitude of the MRR in m.a.s.l.
    chi : float, optional
        Threshold between 0 and 1 to remove the time records with less than chi % from the analysis. The default is 0.7.
    r : float, optional
        Standard deviation parameter used to remove outliers from the average signal. 
        A low r value (e.g. 0.8) will remove most of values whereas a high r value (e.g. 2) will keep more values for the 
        following smoothing. To be use with caution. The default is 1.5.
    sigma : float, optional
        Standard deviation used for the gaussian filtering along time dimension.
        Low sigma value (e.g. 5) will keep more signal but less smooth.
        High sigma value (e.g. 20) will remove larger part of signal and smooth it. The default is 20.
    limit_interpol : float, optional
        Maximum duration in minutes between 2 values where the interpolation might be done. The default is 90.

    Returns
    -------
    hmaxZe_smooth : masked array
        Array of melting layer height elevation in m.a.g.l according the Ze refletivity factor method, along time dimension.
    hmingradZe_smooth : masked array
        Array of melting layer top elevation in m.a.g.l according the Ze refletivity factor method, along time dimension.
    hmaxgradZe_smooth : masked array
        Array of melting layer bottom elevation in m.a.g.l according the Ze refletivity factor method, along time dimension.
    h_smooth : masked array
        Array of melting layer height elevation in m.a.g.l according the W Doppler velocity method, along time dimension.
    hmaxgrad2W_smooth : masked array
        Array of melting layer top elevation in m.a.g.l according the W Doppler velocity method, along time dimension.
    hmingrad2W_smooth : masked array
        Array of melting layer bottom elevation in m.a.g.l according the W Doppler velocity method, along time dimension.
    time : masked array
        Array of the time records from event_start to event_stop
         
    """
    
    import netCDF4 as nc
    import numpy as np
    #import time
    import pandas as pd
    import numpy.ma as ma
    import matplotlib.dates as mdate
    import matplotlib.pyplot as plt
    from datetime import datetime
    from scipy import stats
    from scipy.ndimage import gaussian_filter1d
    import astropy.convolution.convolve
    from astropy.convolution import Gaussian2DKernel

    ds=nc.Dataset(MRRncfile, mode='r')
    #print(ds.variables.keys())
    
    event_date = event_start[:10]
    
    time=ds.variables['time']
    ind_start = np.where(event_start == pd.to_datetime(time[:].data, unit='s'))[0][0]
    ind_stop = np.where(event_stop == pd.to_datetime(time[:].data, unit='s'))[0][0]
    time= time[ind_start:ind_stop]
    Ze= ds.variables['Ze'][ind_start:ind_stop] #reflectivity factor
    W=ds.variables['W'][ind_start:ind_stop]
    height= ds.variables['height'][ind_start:ind_stop] #elevation above MRR
    
    ds.close()

    #% filter the reflectivity
    
    for i in range(0,len(Ze[:])):
        #if more than 30% of the vertical bins are NaN --> reject all the values for the time step (i.e set Filter=False)
        if ma.count_masked(Ze[i,:])/len(Ze[i,:]) > chi: #ma.count_masked(gradW[i,:])
           #W_filt[i,:]= ma.masked_where(ma.count_masked(W_filt[i,:])/len(W_filt[i,:]) > 0.5,W_filt[i,:])
           Ze[i,:]= ma.masked_where(True,Ze[i,:])
           
    #smooth the reflectivity along z-axis before computing the maximum
    k = Gaussian2DKernel(0.5) #smooth only along the z-axis
    Ze_filtered = astropy.convolution.convolve(Ze[:,:], k, boundary='fill')
    Ze_filtered= ma.array(data=Ze_filtered,mask=Ze.mask, fill_value=-9999)
    
    maxZe= ma.MaskedArray.max(Ze_filtered[:,:],axis=1) #we will use his mask to filter values 
    #maxZe= ma.masked_where((maxgradW<0.5)|(maxgradW>2),maxgradW) #only keep the strong gradients
    imaxZe=ma.masked_array(ma.MaskedArray.argmax(Ze_filtered[:,:],axis=1),fill_value=0,
                             mask=maxZe.mask) #array of index of the maxima
    hmaxZe=ma.masked_array(height[:,imaxZe.data][0],mask=imaxZe.mask)
    
    #% outliers filtering
    hmaxZe = ma.masked_where(hmaxZe<400,hmaxZe) #remove the first bin close tothe ground
    
    std_Ze = r
    z_score = np.abs(stats.zscore(ma.filled(hmaxZe,fill_value=np.nan),
                                  nan_policy='omit'))
    print('remove all values above '+ str(hmaxZe.mean()+std_Ze*hmaxZe.std())+
          ' and below '+str(hmaxZe.mean()-std_Ze*hmaxZe.std()))
    hmaxZe = ma.masked_where(z_score>std_Ze,hmaxZe) #set the mask as True where z_score>3
    
    
    #%convert the masked array to pd.df and interpolate the NAN
    hmaxZe_interpol=pd.DataFrame(hmaxZe).interpolate(method='linear', limit=limit_interpol, 
                                                       limit_area='inside')
    hmaxZe_smooth = gaussian_filter1d(hmaxZe_interpol.values[:,0],sigma=sigma)
    hmaxZe_smooth = ma.masked_values(hmaxZe_smooth, np.nan)
    hmaxZe_smooth = ma.masked_array(data=hmaxZe_smooth.data, mask=np.isnan(hmaxZe_smooth.data)) #reconvert to masked array
    
    #%compute gradZe
    gradZe= np.gradient(Ze_filtered, axis=1)
    grad2Ze= np.gradient(gradZe, axis=1)
    
    #find the minimum gradZe
    hML_lim= hmaxZe_smooth.mean() + 500 #do not consider reflectivities 500m above the ML height when searching for the ML top
    mingradZe= ma.MaskedArray.min(ma.masked_where(height>hML_lim,gradZe[:,:]),axis=1) #we will use his mask to filter values 
    #maxgradZe= ma.masked_where((maxgradZe<0.5)|(maxgradZe>2),maxgradZe) #only keep the strong gradients
    imingradZe=ma.masked_array(ma.MaskedArray.argmin(gradZe[:,:],axis=1),fill_value=0,
                             mask=mingradZe.mask) #array of index of the maxima
    
    hmingradZe=ma.masked_array(height[:,imingradZe.data][0],mask=imingradZe.mask)
    
    #%
    hmingradZe = ma.masked_where(hmingradZe<400,hmingradZe)
    hmingradZe = ma.masked_where(hmingradZe>(hmaxZe_smooth.mean()+500),hmingradZe) #restrict the search of MLtop to 500m above the MLheight
    #hmingradZe = ma.masked_values(hmingradZe, hmaxZe_smooth) #masked the values of min grad where maxZe does not exist
    std_gradZe = r
    z_score = np.abs(stats.zscore(ma.filled(hmingradZe,fill_value=np.nan),
                                  nan_policy='omit'))
    print('remove all values above '+ str(hmingradZe.mean()+std_gradZe*hmingradZe.std())+
          ' and below '+str(hmingradZe.mean()-std_gradZe*hmingradZe.std()))
    
    hmingradZe = ma.masked_where(z_score>std_gradZe,hmingradZe) #set the mask as True where z_score>3
    
    #%
    hmingradZe_interpol=pd.DataFrame(hmingradZe).interpolate(method='linear', limit=limit_interpol, 
                                                       limit_area='inside')
    hmingradZe_smooth = gaussian_filter1d(hmingradZe_interpol.values[:,0],sigma=sigma)
    hmingradZe_smooth = ma.masked_values(hmingradZe_smooth,  np.nan)
    hmingradZe_smooth = ma.masked_array(data=hmingradZe_smooth.data, 
                                        mask=np.isnan(hmingradZe_smooth.data)) #reconvert to masked array
    
    #% find the max gradZe (=== the bottom of the ML)
    
    #the max of gradient is searched for the values below the position of the peak of reflectivity only
    gradZe_formax = gradZe
    for i in range(0,len(time)):
        gradZe_formax[i,:] = ma.masked_where(np.arange(0,31)>=imaxZe[i],gradZe[i,:])
        
    maxgradZe= ma.MaskedArray.max(gradZe_formax[:,:],axis=1) #we will use his mask to filter values 
    #maxgradZe= ma.masked_where((maxgradZe<0.5)|(maxgradZe>2),maxgradZe) #only keep the strong gradients
    imaxgradZe=ma.masked_array(ma.MaskedArray.argmax(gradZe_formax[:,:],axis=1),fill_value=0,
                             mask=maxgradZe.mask) #array of index of the maxima
    
    hmaxgradZe=ma.masked_array(height[:,imaxgradZe.data][0],mask=imaxgradZe.mask)
    
    #%
    hmaxgradZe = ma.masked_where(hmaxgradZe<400,hmaxgradZe)
    hmaxgradZe = ma.masked_where(hmaxgradZe<(hmaxZe_smooth.mean()-500),hmaxgradZe)
    hmaxgradZe = ma.masked_where(hmaxgradZe>hmaxZe_smooth, hmaxgradZe) #masked the values of max grad where maxZe is above maxZe
    from scipy import stats
    std_gradZe = r
    z_score = np.abs(stats.zscore(ma.filled(hmaxgradZe,fill_value=np.nan),
                                  nan_policy='omit'))
    print('remove all values above '+ str(hmaxgradZe.mean()+std_gradZe*hmaxgradZe.std())+
          ' and below '+str(hmaxgradZe.mean()-std_gradZe*hmaxgradZe.std()))
    
    hmaxgradZe = ma.masked_where(z_score>std_gradZe,hmaxgradZe) #set the mask as True where z_score>3
    #%
    hmaxgradZe_interpol=pd.DataFrame(hmaxgradZe).interpolate(method='linear', limit=limit_interpol, 
                                                       limit_area='inside')
    #reject the values of the gradient elevation (set as NAN) if above the max reflectivity
    hmaxgradZe_interpol[hmaxgradZe_interpol >= hmaxZe_interpol] = np.nan
    hmaxgradZe_smooth = gaussian_filter1d(hmaxgradZe_interpol.values[:,0],sigma=sigma)
    hmaxgradZe_smooth = ma.masked_values(hmaxgradZe_smooth, np.nan)
    hmaxgradZe_smooth = ma.masked_array(data=hmaxgradZe_smooth.data, 
                                        mask=np.isnan(hmaxgradZe_smooth.data)) #reconvert to masked array
    
    
    #% ML detection from Doppler velocity W
    
    #Filter the velocity
    W_filt=ma.masked_outside(W,-6,25)
    
    for i in range(0,len(W_filt[:])):
        #if more than chi% of the vertical bins are NaN --> reject all the values for the time step (i.e set Filter=False)
        if ma.count_masked(W_filt[i,:])/len(W_filt[i,:]) > chi: #ma.count_masked(gradW[i,:])
           #W_filt[i,:]= ma.masked_where(ma.count_masked(W_filt[i,:])/len(W_filt[i,:]) > 0.5,W_filt[i,:])
           W_filt[i,:]= ma.masked_where(True,W_filt[i,:])
    
    #smooth the velocity along z-axis before computing the gradient
    k = Gaussian2DKernel(0.5)
    z_filtered = astropy.convolution.convolve(W_filt[:,:], k, boundary='fill')
    z_filtered= ma.array(data=z_filtered,mask=W_filt.mask, fill_value=-9999)
    gradW= -np.gradient(z_filtered, axis=1)
    
    maxgradW= ma.MaskedArray.max(gradW[:,:],axis=1) #we will use his mask to filter values 
    maxgradW= ma.masked_where((maxgradW<0.25)|(maxgradW>4),maxgradW) #only keep the strong gradients
    imaxgrad=ma.masked_array(ma.MaskedArray.argmax(gradW[:,:],axis=1),fill_value=0,
                             mask=maxgradW.mask) #array of index of the maxima
    
    hmaxgrad=ma.masked_array(height[:,imaxgrad.data][0],mask=imaxgrad.mask)
    
    #%
    std_gradW = r
    z_score = np.abs(stats.zscore(ma.filled(hmaxgrad,fill_value=np.nan),
                                  nan_policy='omit'))
    print('remove all values above '+ str(hmaxgrad.mean()+std_gradW*hmaxgrad.std())+
          ' and below '+str(hmaxgrad.mean()-std_gradW*hmaxgrad.std()))
    
    hmaxgrad = ma.masked_where(z_score>std_gradW,hmaxgrad) #set the mask as True where z_score>3
    
    #%
    h_interpol=pd.DataFrame(hmaxgrad).interpolate(method='linear', limit=limit_interpol, 
                                                       limit_area='inside')
    h_smooth = gaussian_filter1d(h_interpol.values[:,0],sigma=sigma)
    h_smooth = ma.masked_values(h_smooth,  np.nan)
    h_smooth = ma.masked_array(data=h_smooth.data, mask=np.isnan(h_smooth.data)) #reconvert to masked array
    
    #%## compute grad2W
    grad2W= -np.gradient(gradW, axis=1)
    
    ## find the max of grad2W
    hML_lim= h_smooth.mean() + 500 #do not consider velocities 500m above the ML height when searching for the ML top
    maxgrad2W= ma.MaskedArray.max(ma.masked_where(height>hML_lim,grad2W),axis=1) #we will use his mask to filter values 
    
    maxgrad2W= ma.masked_where((maxgrad2W<0.25)|(maxgrad2W>3),maxgrad2W) #only keep the strong gradients
    imaxgrad2W=ma.masked_array(ma.MaskedArray.argmax(
        ma.masked_where(height>hML_lim,grad2W),axis=1),
        fill_value=0, mask=maxgrad2W.mask) #array of index of the maxima
    hmaxgrad2W=ma.masked_array(height[:,imaxgrad2W.data][0],mask=imaxgrad2W.mask)
    
    hmaxgrad2W = ma.masked_values(hmaxgrad2W, h_smooth) #masked the values of min grad where maxZe does not exist
    #%filter the maximum values of the gradient which are outside mean+/- X std
    std_grad2W = r
    z_score2 = np.abs(stats.zscore(ma.filled(hmaxgrad2W,fill_value=np.nan),
                                  nan_policy='omit'))
    print('remove all values above '+ str(hmaxgrad2W.mean()+std_grad2W*hmaxgrad2W.std())+
          ' and below '+str(hmaxgrad2W.mean()-std_grad2W*hmaxgrad2W.std()))
    
    hmaxgrad2W = ma.masked_where(z_score2 > std_grad2W, hmaxgrad2W) #set the mask as True where z_score>3
    
    #reject the values of the second derivative where the values of the first derivative where rejected,
    #i.e. the second gradient maximum can exist only if the max of first derivative exists
    #hmaxgrad2W = ma.masked_where(maxgradW.mask==True, hmaxgrad2W)
    
    #%convert the masked array to pd.df and interpolate the NAN
    hmaxgrad2W_interpol=pd.DataFrame(hmaxgrad2W).interpolate(method='linear', limit=limit_interpol, 
                                                       limit_area='inside')
    hmaxgrad2W_smooth = gaussian_filter1d(hmaxgrad2W_interpol.values[:,0],sigma=sigma)
    hmaxgrad2W_smooth = ma.masked_values(hmaxgrad2W_smooth,  np.nan)
    hmaxgrad2W_smooth = ma.masked_array(data=hmaxgrad2W_smooth.data, mask=np.isnan(hmaxgrad2W_smooth.data)) #reconvert to masked array
    hmaxgrad2W_smooth = ma.masked_where(hmaxgrad2W_smooth<h_smooth, hmaxgrad2W_smooth) #top of ML can't be lower than the height of ML
    
    #%# find the min of grad2
    mingrad2W= ma.MaskedArray.min(ma.masked_where(height>hML_lim,grad2W),axis=1) #we will use his mask to filter values 
    
    mingrad2W= ma.masked_where((np.abs(mingrad2W)<0.2) |
                               (np.abs(mingrad2W)>3),mingrad2W) #only keep the strong gradients
    imingrad2W=ma.masked_array(
        ma.MaskedArray.argmin(ma.masked_where(height>hML_lim,grad2W),axis=1),
        fill_value=0, mask=mingrad2W.mask) #array of index of the maxima
    hmingrad2W= ma.masked_array(height[:,imingrad2W.data][0],mask=imingrad2W.mask)
    hmingrad2W= ma.masked_where(hmingrad2W<=500, hmingrad2W) #minimum can't be less or equal to 500m (because it is the boundary) 
    
    hmingrad2W = ma.masked_values(hmingrad2W, h_smooth) #masked the values of min grad2 where maxgrad does not exist
    #%filter the maximum values of the gradient which are outside mean+/-3std
    std_grad2W = r
    z_score3 = np.abs(stats.zscore(ma.filled(hmingrad2W,fill_value=np.nan),
                                  nan_policy='omit'))
    print('remove all values above '+ str(hmingrad2W.mean()+std_grad2W*hmingrad2W.std())+
          ' and below '+str(hmingrad2W.mean()-std_grad2W*hmingrad2W.std()))
    
    hmingrad2W = ma.masked_where(z_score3 > std_grad2W, hmingrad2W) #set the mask as True where z_score>3
    
    #reject the values of the second derivative where the values of the first derivative where rejected,
    #i.e. the second gradient maximum can exist only if the max of first derivative exists
    #hmingrad2W = ma.masked_where(maxgradW.mask==True, hmingrad2W)
    
    #remove the values wich have an elevation above are equal to those of the first derivative
    hmingrad2W = ma.masked_where(hmingrad2W >= h_smooth, hmingrad2W)
    #hmingrad2W = ma.masked_where(h_smooth.mask, hmingrad2W)
    
    #%convert the masked array to pd.df and interpolate the NAN
    hmingrad2W_interpol=pd.DataFrame(hmingrad2W).interpolate(method='linear', limit=limit_interpol, 
                                                       limit_area='inside')
    hmingrad2W_smooth = gaussian_filter1d(hmingrad2W_interpol.values[:,0],sigma=sigma)
    hmingrad2W_smooth = ma.masked_values(hmingrad2W_smooth, np.nan)
    hmingrad2W_smooth = ma.masked_array(data=hmingrad2W_smooth.data, mask=np.isnan(hmingrad2W_smooth.data)) #reconvert to masked array#reconvert to masked array
    hmingrad2W_smooth = ma.masked_where(hmingrad2W_smooth>h_smooth, hmingrad2W_smooth) #ML bottom cant be above ML height

    return hmaxZe_smooth,hmingradZe_smooth,hmaxgradZe_smooth,h_smooth,hmaxgrad2W_smooth,hmingrad2W_smooth, time


def plot_MRR_melting_layer(ML_Ze_height, ML_Ze_top, ML_Ze_bot, ML_W_height, 
                          ML_W_top, ML_W_bot, time, 
                          event_start, event_stop, alt_station, path_dir_out):
    
    """
    plot and save the melting layer elevation
    """
    
    import matplotlib.dates as mdate
    import matplotlib.pyplot as plt
    import pandas as pd
    
    event_date= event_start[:10]
    
    #Plot melting layer from Ze reflectivity factor
    plt.plot_date(pd.to_datetime(time[:].data, unit='s'),
                  ML_Ze_height, '-', label=r'max(Ze)',
                  markersize=3, markeredgewidth=1)
    plt.plot_date(pd.to_datetime(time[:].data, unit='s'),
                  ML_Ze_top, '-', label=r'min(gradZe)',
                  markersize=2, markeredgewidth=1)
    plt.plot_date(pd.to_datetime(time[:].data, unit='s'),
                  ML_Ze_bot, '-', label=r'max(gradZe)',
                  markersize=2, markeredgewidth=1)
    plt.ylim(0, 3000)
    plt.xlim(pd.to_datetime(time[0], unit='s'),pd.to_datetime(time[-1], unit='s'))
    plt.ylabel('Height (m.a.g.l)')
    plt.xlabel('Time [UTC]')
    plt.title('MRR OSUG-B ' +event_date + '\n ML elevation from Ze reflectivity factor')
    plt.legend(fontsize=10)
    plt.grid(which='major')
    plt.grid(which='minor', alpha=0.5)
    plt.minorticks_on()
    plt.gcf().autofmt_xdate()
    date_format = mdate.DateFormatter('%d/%m %H:00')
    plt.gca().xaxis.set_major_formatter(date_format)
    #plt.tight_layout()
    plt.rc('xtick', labelsize=12) 
    plt.savefig(path_dir_out+event_date+'_Ze_OSUG-B.png', dpi=300)
    plt.show()
    
    #Plot Melting layer according W Doppler velocity
    plt.plot_date(pd.to_datetime(time[:].data, unit='s'),
                  ML_W_height, '-', label=r'max($\nabla W$)',
                  markersize=3, markeredgewidth=1)
    plt.plot_date(pd.to_datetime(time[:].data, unit='s'),
                  ML_W_top, '-', label=r'max($\nabla^{2} W$)',
                  markersize=2, markeredgewidth=1)
    plt.plot_date(pd.to_datetime(time[:].data, unit='s'),
                  ML_W_bot, '-', label=r'min($\nabla^{2} W$)',
                  markersize=2, markeredgewidth=1)
    plt.ylim(0, 3000)
    plt.xlim(pd.to_datetime(time[0], unit='s'),pd.to_datetime(time[-1], unit='s'))
    plt.ylabel('Height [m.a.g.l]')
    plt.xlabel('Time [UTC]')
    plt.title('MRR OSUG-B '+event_date+'\n ML elevation from W Doppler velocity')
    plt.legend(fontsize=10)
    plt.grid(which='major')
    plt.grid(which='minor', alpha=0.5)
    plt.minorticks_on()
    plt.gcf().autofmt_xdate()
    date_format = mdate.DateFormatter('%H:00')
    plt.gca().xaxis.set_major_formatter(date_format)
    #plt.tight_layout()
    plt.rc('xtick', labelsize=12) 
    plt.savefig(path_dir_out+event_date+'_W_OSUG-B.png', dpi=300)
    plt.show()
    
    #Plot melting layer from both Ze and W in meters above sea level
    plt.figure(facecolor='white', figsize=(8,5))
    plt.plot_date(pd.to_datetime(time[:].data, unit='s'),
                  ML_W_height + alt_station, '-', label=r'$max(\partial_z{W})$',
                  markersize=1, markeredgewidth=1, alpha=0.9)
    plt.plot_date(pd.to_datetime(time[:].data, unit='s'),
                  ML_W_top + alt_station, '-', label=r'$max(\partial_z^2{W})$',
                  markersize=1, markeredgewidth=1, alpha=0.9)
    plt.plot_date(pd.to_datetime(time[:].data, unit='s'),
                  ML_W_bot + alt_station, '-', label=r'$min(\partial_z{W})$',
                  markersize=1, markeredgewidth=1, alpha=0.9)
    plt.plot_date(pd.to_datetime(time[:].data, unit='s'),
                  ML_Ze_height + alt_station, '--', label=r'$max(Ze)$', color='tab:blue',
                  markersize=1, markeredgewidth=1, alpha=0.9)
    plt.plot_date(pd.to_datetime(time[:].data, unit='s'),
                  ML_Ze_top + alt_station, '--', label=r'$min(\partial_z{Ze})$', color='tab:orange',
                  markersize=1, markeredgewidth=1, alpha=0.9)
    plt.plot_date(pd.to_datetime(time[:].data, unit='s'),
                  ML_Ze_bot + alt_station, '--', label=r'$max(\partial_z{Ze})$', color='tab:green',
                  markersize=1, markeredgewidth=1, alpha=0.9)
    plt.ylim(0, 3000)
    plt.xlim(pd.to_datetime(time[0], unit='s'),pd.to_datetime(time[-1], unit='s'))
    plt.ylabel('Elevation [m.a.s.l]')
    plt.xlabel('Time [UTC]')
    plt.title('MRR OSUG-B '+event_date+'\n Melting layer elevation')
    plt.legend(fontsize=10, ncol=2)
    plt.grid(which='minor', alpha=0.1)
    plt.grid(which='major', alpha=0.3)
    plt.minorticks_on()
    plt.gcf().autofmt_xdate()
    date_format = mdate.DateFormatter('%d%b \n %H:%M')
    plt.gca().xaxis.set_major_formatter(date_format)
    plt.tight_layout()
    plt.rc('xtick', labelsize=10) 
    plt.savefig(path_dir_out+event_date+'_ML_OSUG-B.png', dpi=300)
    plt.show()
    
def plot_spec_time(spec_time, MRRncfile, event_start, event_stop, path_dir_out, 
                   chi=0.7, r=1.5, sigma=20, limit_interpol=90):
    """
    Plot the vertical profiles of reflectivity factor and Doppler velocity at a specific time.
    Also plot in dashed lines the melting layers boundaries if detected by the algorithm.

    Parameters
    ----------
    spec_time : string
        Specific time [UTC] for which the vertical profile is wanted, in format %Y-%m-%d %H:%M:%S.
    MRRncfile : netcdf
        MRR netcdf file.
    event_start : string
        Starting time [UTC] of the event in format %Y-%m-%d %H:%M:%S.
    event_stop : string
        Ending time [UTC] of the event in format %Y-%m-%d %H:%M:%S.
    path_dir_out : string
        Path of the directory to save the plots.
    chi : float, optional
        Threshold between 0 and 1 to remove the time records with less than chi % from the analysis. The default is 0.7.
    r : float, optional
        Standard deviation parameter used to remove outliers from the average signal. 
        A low r value (e.g. 0.8) will remove most of values whereas a high r value (e.g. 2) will keep more values for the 
        following smoothing. To be use with caution. The default is 1.5.
    sigma : float, optional
        Standard deviation used for the gaussian filtering along time dimension.
        Low sigma value (e.g. 5) will keep more signal but less smooth.
        High sigma value (e.g. 20) will remove larger part of signal and smooth it. 
        The default is 20.
    limit_interpol : float, optional
        Maximum duration in minutes between 2 values where the interpolation might be done. The default is 90.

    Returns
    -------
    None.

    """
    
    import netCDF4 as nc
    import numpy as np
    #import time
    import pandas as pd
    import numpy.ma as ma
    import matplotlib.dates as mdate
    import matplotlib.pyplot as plt
    from datetime import datetime
    from scipy import stats
    from scipy.ndimage import gaussian_filter1d
    import astropy.convolution.convolve
    from astropy.convolution import Gaussian2DKernel
    
    ds=nc.Dataset(MRRncfile, mode='r')
    #print(ds.variables.keys())
    
    event_date = event_start[:10]
    
    time=ds.variables['time']
    ind_start = np.where(event_start == pd.to_datetime(time[:].data, unit='s'))[0][0]
    ind_stop = np.where(event_stop == pd.to_datetime(time[:].data, unit='s'))[0][0]
    time= time[ind_start:ind_stop]
    Ze= ds.variables['Ze'][ind_start:ind_stop] #reflectivity factor
    W=ds.variables['W'][ind_start:ind_stop]
    height= ds.variables['height'][ind_start:ind_stop] #elevation above MRR
    
    ds.close()
    
    #% filter the reflectivity
    
    for i in range(0,len(Ze[:])):
        #if more than 30% of the vertical bins are NaN --> reject all the values for the time step (i.e set Filter=False)
        if ma.count_masked(Ze[i,:])/len(Ze[i,:]) > chi: #ma.count_masked(gradW[i,:])
           #W_filt[i,:]= ma.masked_where(ma.count_masked(W_filt[i,:])/len(W_filt[i,:]) > 0.5,W_filt[i,:])
           Ze[i,:]= ma.masked_where(True,Ze[i,:])
           
    #smooth the reflectivity along z-axis before computing the maximum
    k = Gaussian2DKernel(0.5) #smooth only along the z-axis
    Ze_filtered = astropy.convolution.convolve(Ze[:,:], k, boundary='fill')
    Ze_filtered= ma.array(data=Ze_filtered,mask=Ze.mask, fill_value=-9999)
    
    maxZe= ma.MaskedArray.max(Ze_filtered[:,:],axis=1) #we will use his mask to filter values 
    #maxZe= ma.masked_where((maxgradW<0.5)|(maxgradW>2),maxgradW) #only keep the strong gradients
    imaxZe=ma.masked_array(ma.MaskedArray.argmax(Ze_filtered[:,:],axis=1),fill_value=0,
                             mask=maxZe.mask) #array of index of the maxima
    hmaxZe=ma.masked_array(height[:,imaxZe.data][0],mask=imaxZe.mask)
    
    #% outliers filtering
    hmaxZe = ma.masked_where(hmaxZe<400,hmaxZe) #remove the first bin close tothe ground
    
    std_Ze = r
    z_score = np.abs(stats.zscore(ma.filled(hmaxZe,fill_value=np.nan),
                                  nan_policy='omit'))
    print('remove all values above '+ str(hmaxZe.mean()+std_Ze*hmaxZe.std())+
          ' and below '+str(hmaxZe.mean()-std_Ze*hmaxZe.std()))
    hmaxZe = ma.masked_where(z_score>std_Ze,hmaxZe) #set the mask as True where z_score>3
    
    
    #%convert the masked array to pd.df and interpolate the NAN
    hmaxZe_interpol=pd.DataFrame(hmaxZe).interpolate(method='linear', limit=limit_interpol, 
                                                       limit_area='inside')
    hmaxZe_smooth = gaussian_filter1d(hmaxZe_interpol.values[:,0],sigma=sigma)
    hmaxZe_smooth = ma.masked_values(hmaxZe_smooth, np.nan)
    hmaxZe_smooth = ma.masked_array(data=hmaxZe_smooth.data, mask=np.isnan(hmaxZe_smooth.data)) #reconvert to masked array
    
    #%compute gradZe
    gradZe= np.gradient(Ze_filtered, axis=1)
    grad2Ze= np.gradient(gradZe, axis=1)
    
    #find the minimum gradZe
    hML_lim= hmaxZe_smooth.mean() + 500 #do not consider reflectivities 500m above the ML height when searching for the ML top
    mingradZe= ma.MaskedArray.min(ma.masked_where(height>hML_lim,gradZe[:,:]),axis=1) #we will use his mask to filter values 
    #maxgradZe= ma.masked_where((maxgradZe<0.5)|(maxgradZe>2),maxgradZe) #only keep the strong gradients
    imingradZe=ma.masked_array(ma.MaskedArray.argmin(gradZe[:,:],axis=1),fill_value=0,
                             mask=mingradZe.mask) #array of index of the maxima
    
    hmingradZe=ma.masked_array(height[:,imingradZe.data][0],mask=imingradZe.mask)
    
    #%
    hmingradZe = ma.masked_where(hmingradZe<400,hmingradZe)
    hmingradZe = ma.masked_where(hmingradZe>(hmaxZe_smooth.mean()+500),hmingradZe) #restrict the search of MLtop to 500m above the MLheight
    #hmingradZe = ma.masked_values(hmingradZe, hmaxZe_smooth) #masked the values of min grad where maxZe does not exist
    std_gradZe = r
    z_score = np.abs(stats.zscore(ma.filled(hmingradZe,fill_value=np.nan),
                                  nan_policy='omit'))
    print('remove all values above '+ str(hmingradZe.mean()+std_gradZe*hmingradZe.std())+
          ' and below '+str(hmingradZe.mean()-std_gradZe*hmingradZe.std()))
    
    hmingradZe = ma.masked_where(z_score>std_gradZe,hmingradZe) #set the mask as True where z_score>3
    
    #%
    hmingradZe_interpol=pd.DataFrame(hmingradZe).interpolate(method='linear', limit=limit_interpol, 
                                                       limit_area='inside')
    hmingradZe_smooth = gaussian_filter1d(hmingradZe_interpol.values[:,0],sigma=sigma)
    hmingradZe_smooth = ma.masked_values(hmingradZe_smooth,  np.nan)
    hmingradZe_smooth = ma.masked_array(data=hmingradZe_smooth.data, 
                                        mask=np.isnan(hmingradZe_smooth.data)) #reconvert to masked array
    
    #% find the max gradZe (=== the bottom of the ML)
    
    #the max of gradient is searched for the values below the position of the peak of reflectivity only
    gradZe_formax = gradZe
    for i in range(0,len(time)):
        gradZe_formax[i,:] = ma.masked_where(np.arange(0,31)>=imaxZe[i],gradZe[i,:])
        
    maxgradZe= ma.MaskedArray.max(gradZe_formax[:,:],axis=1) #we will use his mask to filter values 
    #maxgradZe= ma.masked_where((maxgradZe<0.5)|(maxgradZe>2),maxgradZe) #only keep the strong gradients
    imaxgradZe=ma.masked_array(ma.MaskedArray.argmax(gradZe_formax[:,:],axis=1),fill_value=0,
                             mask=maxgradZe.mask) #array of index of the maxima
    
    hmaxgradZe=ma.masked_array(height[:,imaxgradZe.data][0],mask=imaxgradZe.mask)
    
    #%
    hmaxgradZe = ma.masked_where(hmaxgradZe<400,hmaxgradZe)
    hmaxgradZe = ma.masked_where(hmaxgradZe<(hmaxZe_smooth.mean()-500),hmaxgradZe)
    hmaxgradZe = ma.masked_where(hmaxgradZe>hmaxZe_smooth, hmaxgradZe) #masked the values of max grad where maxZe is above maxZe
    from scipy import stats
    std_gradZe = r
    z_score = np.abs(stats.zscore(ma.filled(hmaxgradZe,fill_value=np.nan),
                                  nan_policy='omit'))
    print('remove all values above '+ str(hmaxgradZe.mean()+std_gradZe*hmaxgradZe.std())+
          ' and below '+str(hmaxgradZe.mean()-std_gradZe*hmaxgradZe.std()))
    
    hmaxgradZe = ma.masked_where(z_score>std_gradZe,hmaxgradZe) #set the mask as True where z_score>3
    #%
    hmaxgradZe_interpol=pd.DataFrame(hmaxgradZe).interpolate(method='linear', limit=limit_interpol, 
                                                       limit_area='inside')
    #reject the values of the gradient elevation (set as NAN) if above the max reflectivity
    hmaxgradZe_interpol[hmaxgradZe_interpol >= hmaxZe_interpol] = np.nan
    hmaxgradZe_smooth = gaussian_filter1d(hmaxgradZe_interpol.values[:,0],sigma=sigma)
    hmaxgradZe_smooth = ma.masked_values(hmaxgradZe_smooth, np.nan)
    hmaxgradZe_smooth = ma.masked_array(data=hmaxgradZe_smooth.data, 
                                        mask=np.isnan(hmaxgradZe_smooth.data)) #reconvert to masked array
    
    
    #% ML detection from Doppler velocity W
    
    #Filter the velocity
    W_filt=ma.masked_outside(W,-6,25)
    
    for i in range(0,len(W_filt[:])):
        #if more than chi% of the vertical bins are NaN --> reject all the values for the time step (i.e set Filter=False)
        if ma.count_masked(W_filt[i,:])/len(W_filt[i,:]) > chi: #ma.count_masked(gradW[i,:])
           #W_filt[i,:]= ma.masked_where(ma.count_masked(W_filt[i,:])/len(W_filt[i,:]) > 0.5,W_filt[i,:])
           W_filt[i,:]= ma.masked_where(True,W_filt[i,:])
    
    #smooth the velocity along z-axis before computing the gradient
    k = Gaussian2DKernel(0.5)
    z_filtered = astropy.convolution.convolve(W_filt[:,:], k, boundary='fill')
    z_filtered= ma.array(data=z_filtered,mask=W_filt.mask, fill_value=-9999)
    gradW= -np.gradient(z_filtered, axis=1)
    
    maxgradW= ma.MaskedArray.max(gradW[:,:],axis=1) #we will use his mask to filter values 
    maxgradW= ma.masked_where((maxgradW<0.25)|(maxgradW>4),maxgradW) #only keep the strong gradients
    imaxgrad=ma.masked_array(ma.MaskedArray.argmax(gradW[:,:],axis=1),fill_value=0,
                             mask=maxgradW.mask) #array of index of the maxima
    
    hmaxgrad=ma.masked_array(height[:,imaxgrad.data][0],mask=imaxgrad.mask)
    
    #%
    std_gradW = r
    z_score = np.abs(stats.zscore(ma.filled(hmaxgrad,fill_value=np.nan),
                                  nan_policy='omit'))
    print('remove all values above '+ str(hmaxgrad.mean()+std_gradW*hmaxgrad.std())+
          ' and below '+str(hmaxgrad.mean()-std_gradW*hmaxgrad.std()))
    
    hmaxgrad = ma.masked_where(z_score>std_gradW,hmaxgrad) #set the mask as True where z_score>3
    
    #%
    h_interpol=pd.DataFrame(hmaxgrad).interpolate(method='linear', limit=limit_interpol, 
                                                       limit_area='inside')
    h_smooth = gaussian_filter1d(h_interpol.values[:,0],sigma=sigma)
    h_smooth = ma.masked_values(h_smooth,  np.nan)
    h_smooth = ma.masked_array(data=h_smooth.data, mask=np.isnan(h_smooth.data)) #reconvert to masked array
    
    #%## compute grad2W
    grad2W= -np.gradient(gradW, axis=1)
    
    ## find the max of grad2W
    hML_lim= h_smooth.mean() + 500 #do not consider velocities 500m above the ML height when searching for the ML top
    maxgrad2W= ma.MaskedArray.max(ma.masked_where(height>hML_lim,grad2W),axis=1) #we will use his mask to filter values 
    
    maxgrad2W= ma.masked_where((maxgrad2W<0.25)|(maxgrad2W>3),maxgrad2W) #only keep the strong gradients
    imaxgrad2W=ma.masked_array(ma.MaskedArray.argmax(
        ma.masked_where(height>hML_lim,grad2W),axis=1),
        fill_value=0, mask=maxgrad2W.mask) #array of index of the maxima
    hmaxgrad2W=ma.masked_array(height[:,imaxgrad2W.data][0],mask=imaxgrad2W.mask)
    
    hmaxgrad2W = ma.masked_values(hmaxgrad2W, h_smooth) #masked the values of min grad where maxZe does not exist
    #%filter the maximum values of the gradient which are outside mean+/- X std
    std_grad2W = r
    z_score2 = np.abs(stats.zscore(ma.filled(hmaxgrad2W,fill_value=np.nan),
                                  nan_policy='omit'))
    print('remove all values above '+ str(hmaxgrad2W.mean()+std_grad2W*hmaxgrad2W.std())+
          ' and below '+str(hmaxgrad2W.mean()-std_grad2W*hmaxgrad2W.std()))
    
    hmaxgrad2W = ma.masked_where(z_score2 > std_grad2W, hmaxgrad2W) #set the mask as True where z_score>3
    
    #reject the values of the second derivative where the values of the first derivative where rejected,
    #i.e. the second gradient maximum can exist only if the max of first derivative exists
    #hmaxgrad2W = ma.masked_where(maxgradW.mask==True, hmaxgrad2W)
    
    #%convert the masked array to pd.df and interpolate the NAN
    hmaxgrad2W_interpol=pd.DataFrame(hmaxgrad2W).interpolate(method='linear', limit=limit_interpol, 
                                                       limit_area='inside')
    hmaxgrad2W_smooth = gaussian_filter1d(hmaxgrad2W_interpol.values[:,0],sigma=sigma)
    hmaxgrad2W_smooth = ma.masked_values(hmaxgrad2W_smooth,  np.nan)
    hmaxgrad2W_smooth = ma.masked_array(data=hmaxgrad2W_smooth.data, mask=np.isnan(hmaxgrad2W_smooth.data)) #reconvert to masked array
    hmaxgrad2W_smooth = ma.masked_where(hmaxgrad2W_smooth<h_smooth, hmaxgrad2W_smooth) #top of ML can't be lower than the height of ML
    
    #%# find the min of grad2
    mingrad2W= ma.MaskedArray.min(ma.masked_where(height>hML_lim,grad2W),axis=1) #we will use his mask to filter values 
    
    mingrad2W= ma.masked_where((np.abs(mingrad2W)<0.2) |
                               (np.abs(mingrad2W)>3),mingrad2W) #only keep the strong gradients
    imingrad2W=ma.masked_array(
        ma.MaskedArray.argmin(ma.masked_where(height>hML_lim,grad2W),axis=1),
        fill_value=0, mask=mingrad2W.mask) #array of index of the maxima
    hmingrad2W= ma.masked_array(height[:,imingrad2W.data][0],mask=imingrad2W.mask)
    hmingrad2W= ma.masked_where(hmingrad2W<=500, hmingrad2W) #minimum can't be less or equal to 500m (because it is the boundary) 
    
    hmingrad2W = ma.masked_values(hmingrad2W, h_smooth) #masked the values of min grad2 where maxgrad does not exist
    #%filter the maximum values of the gradient which are outside mean+/-3std
    std_grad2W = r
    z_score3 = np.abs(stats.zscore(ma.filled(hmingrad2W,fill_value=np.nan),
                                  nan_policy='omit'))
    print('remove all values above '+ str(hmingrad2W.mean()+std_grad2W*hmingrad2W.std())+
          ' and below '+str(hmingrad2W.mean()-std_grad2W*hmingrad2W.std()))
    
    hmingrad2W = ma.masked_where(z_score3 > std_grad2W, hmingrad2W) #set the mask as True where z_score>3
    
    #reject the values of the second derivative where the values of the first derivative where rejected,
    #i.e. the second gradient maximum can exist only if the max of first derivative exists
    #hmingrad2W = ma.masked_where(maxgradW.mask==True, hmingrad2W)
    
    #remove the values wich have an elevation above are equal to those of the first derivative
    hmingrad2W = ma.masked_where(hmingrad2W >= h_smooth, hmingrad2W)
    #hmingrad2W = ma.masked_where(h_smooth.mask, hmingrad2W)
    
    #%convert the masked array to pd.df and interpolate the NAN
    hmingrad2W_interpol=pd.DataFrame(hmingrad2W).interpolate(method='linear', limit=limit_interpol, 
                                                       limit_area='inside')
    hmingrad2W_smooth = gaussian_filter1d(hmingrad2W_interpol.values[:,0],sigma=sigma)
    hmingrad2W_smooth = ma.masked_values(hmingrad2W_smooth, np.nan)
    hmingrad2W_smooth = ma.masked_array(data=hmingrad2W_smooth.data, mask=np.isnan(hmingrad2W_smooth.data)) #reconvert to masked array#reconvert to masked array
    hmingrad2W_smooth = ma.masked_where(hmingrad2W_smooth>h_smooth, hmingrad2W_smooth) #ML bottom cant be above ML height

    
    #%plot reflectivity profile at sepecific time (e.g. sat_time)
    id_spec_time = np.argwhere(pd.to_datetime(time[:].data, unit='s')==spec_time)[0][0]
    Ze_avg = Ze_filtered[id_spec_time-5:id_spec_time+5,:].mean(axis=0)

    fig = plt.figure(facecolor='white')
    ax1 = fig.add_subplot(111)
    ax2 = ax1.twiny()
    ax1.set_xlim(0,35)
    ax2.set_xlim(-4,6)
    p1=ax1.plot(Ze_avg, height[id_spec_time,:], '-+',c='b', label='Ze')
    p2=ax2.plot(gradZe[id_spec_time-5:id_spec_time+5,:].mean(axis=0), height[id_spec_time,:], '-+',c='r', label=r'$\partial_{z}{Ze}$')
    ax2.set_xlabel(r"$10^{-2}.dBZ.m^{-1}$", color='r')
    ax2.tick_params(axis='x', labelcolor='r')
    ax1.set_xlabel("dBZ", color='b')
    ax1.tick_params(axis='x', labelcolor='b')
    ax1.set_ylabel("height [m.a.g.l]")
    h1=ax1.axhline(hmaxZe[id_spec_time], ls='--', alpha=0.7, label='max(Ze)')
    h2=ax1.axhline(hmaxgradZe[id_spec_time], ls='--',color='g', alpha=0.7, label=r'max($\partial_{z}{Ze}$)')
    h3=ax1.axhline(hmingradZe[id_spec_time], ls='--',color='orange', alpha=0.7, label=r'min($\partial_{z}{Ze}$)')
    plt.legend(handles=p1+p2+[h1]+[h2]+[h3])
    #labs = [l.get_label() for l in (p1+p2+[h1]+[h2]+[h3])]
    #ax1.legend(p1+p2+[h1]+[h2]+[h3],labs,fontsize=10, loc=0)
    plt.title('MRR_OSUG-B Reflectivity profile\n'+spec_time)
    fig.tight_layout()
    fig.savefig(path_dir_out+'MRR_reflectivity_profile.png',format='png',dpi=300)
    plt.show()
    
    #%plot Doppler velocity profile at sepecific time (e.g. sat_time)
    fig = plt.figure(facecolor='white')
    ax1 = fig.add_subplot(111)
    ax2 = ax1.twiny()
    ax3 = ax1.twiny()
    ax3.spines['top'].set_position(('outward', 30))
    ax3.set_xlim(-1,2)
    ax2.set_xlim(-0.3,2.7)
    ax1.set_xlim(0,6)
    ax1.set_ylim(0,3000)
    p1=ax1.plot(W_filt[id_spec_time-5:id_spec_time+5,:].mean(axis=0), height[id_spec_time,:], '-+',c='b', label='W')
    p2=ax2.plot(gradW[id_spec_time,:], height[id_spec_time,:], '-+',c='r', label=r'$\partial_{z}{W}$')
    p3=ax3.plot(grad2W[id_spec_time,:], height[id_spec_time,:], '-+',c='purple', label=r'$\partial_{z}^2{W}$')
    ax2.set_xlabel(r"$10^{-2}.s^{-1}$", color='r')
    ax2.tick_params(axis='x', labelcolor='r')
    ax1.set_xlabel(r"$m.s^{-1}$", color='b')
    ax1.tick_params(axis='x', labelcolor='b')
    ax3.set_xlabel(r"$10^{-4}.m^{-1}.s^{-1}$", color='purple')
    ax3.tick_params(axis='x', labelcolor='purple')
    ax1.set_ylabel("height [m.a.g.l]")
    h1=ax1.axhline(hmaxgrad[id_spec_time], ls='--', alpha=0.6, label=r'max($\partial_{z}(W)$')
    h2=ax1.axhline(hmaxgrad2W[id_spec_time], ls='--',color='orange', alpha=0.6, label=r'max($\partial_{z}^2{W}$)')
    h3=ax1.axhline(hmingrad2W[id_spec_time], ls='--',color='g', alpha=0.6, label=r'min($\partial_{z}^2{W}$)')
    plt.legend(handles=p1+p2+p3+[h1]+[h2]+[h3], ncol=2)
    #plt.plot(grad2Ze[1499,:], height[1499,:], '-+', label='grad2Ze')
    #labs = [l.get_label() for l in (p1+p2+[h1]+[h2]+[h3])]
    #ax1.legend(p1+p2+[h1]+[h2]+[h3],labs,fontsize=10, loc=0)
    plt.title('MRR_OSUG-B Doppler velocity profile\n'+spec_time)
    fig.tight_layout()
    fig.savefig(path_dir_out+'MRR_Doppler_velocity_profile.png',format='png',dpi=300)
    plt.show()
    
def get_spec_melting_layer(spec_time, ML_Ze_height, ML_Ze_top, ML_Ze_bot,
                           ML_W_height, ML_W_top, ML_W_bot, time,
                           alt_station, avg_time=10):
    """
    

    Parameters
    ----------
    spec_time : TYPE
        DESCRIPTION.
    ML_Ze_height : masked array
        Array of melting layer height elevation in m.a.g.l according the Ze refletivity factor method, along time dimension.
    ML_Ze_top : masked array
        Array of melting layer top elevation in m.a.g.l according the Ze refletivity factor method, along time dimension.
    ML_Ze_bot : masked array
        Array of melting layer bottom elevation in m.a.g.l according the Ze refletivity factor method, along time dimension.
    ML_W_height : masked array
        Array of melting layer height elevation in m.a.g.l according the W Doppler velocity method, along time dimension.
    ML_W_top : masked array
        Array of melting layer top elevation in m.a.g.l according the W Doppler velocity method, along time dimension.
    ML_W_bot : masked array
        Array of melting layer bottom elevation in m.a.g.l according the W Doppler velocity method, along time dimension.
    time : masked array
        Array of the time records of the events
    alt_station : float
        Altitude of the MRR in m.a.s.l.
    avg_time : int, optional
        The meltinng layer values are calculated as the average of the values around the specific time +/- avg_time in minutes. 
        The default is 10.

    Returns
    -------
    BB_MRR : Serie
        Vector of melting layer elevations at the specific time in m.a.s.l.

    """
    
    import pandas as pd
    import numpy as np
    
    id_spec_time = np.where(pd.to_datetime(time[:].data, unit='s') == spec_time)[0][0]
    BB_MRR = pd.Series(dtype='float')
    BB_MRR['heightBB_Ze'] = ML_Ze_height[id_spec_time-avg_time:id_spec_time+avg_time].mean() + alt_station
    BB_MRR['topBB_Ze'] = ML_Ze_top[id_spec_time-avg_time:id_spec_time+avg_time].mean() + alt_station
    BB_MRR['botBB_Ze'] = ML_Ze_bot[id_spec_time-avg_time:id_spec_time+avg_time].mean() + alt_station
    BB_MRR['heightBB_W'] = ML_W_height[id_spec_time-avg_time:id_spec_time+avg_time].mean() + alt_station
    BB_MRR['topBB_W'] = ML_W_top[id_spec_time-avg_time:id_spec_time+avg_time].mean() + alt_station
    BB_MRR['botBB_W'] = ML_W_bot[id_spec_time-avg_time:id_spec_time+avg_time].mean() + alt_station

    return BB_MRR