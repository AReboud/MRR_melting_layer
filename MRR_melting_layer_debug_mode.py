# -*- coding: utf-8 -*-
"""
Created on Sun Jun  5 23:40:18 2022

@author: Arnaud Reboud (IGE)
"""

#Commons
import netCDF4 as nc
import numpy as np
import time
import pandas as pd
import numpy.ma as ma
import matplotlib.dates as mdate
import matplotlib.pyplot as plt
from datetime import datetime
from scipy import stats
from scipy.ndimage import gaussian_filter1d
import astropy.convolution.convolve
from astropy.convolution import Gaussian2DKernel

MRR_file = './Data/MRR/OSUG-B_20201212.nc'
dir_plot = './Plots/MRR/'

ds=nc.Dataset(MRR_file, mode='r')
#print(ds.variables.keys())

sat_time = '2020-12-12 04:00:00'
event_start = '2020-12-12 04:00:00'
event_stop = '2020-12-12 22:00:00'
event_date = event_start[:10]

alt_station = 230 #m.a.s.l : elevation of OSUG-B MRR station

time=ds.variables['time']
ind_start = np.where(event_start == pd.to_datetime(time[:].data, unit='s'))[0][0]
ind_stop = np.where(event_stop == pd.to_datetime(time[:].data, unit='s'))[0][0]
time= time[ind_start:ind_stop]
Ze= ds.variables['Ze'][ind_start:ind_stop] #reflectivity factor
W=ds.variables['W'][ind_start:ind_stop]
height= ds.variables['height'][ind_start:ind_stop] #elevation above MRR

ds.close()
#Parameters
chi = 0.7
r = 1
sigma = 20
time_interp = 90 #min

#%%filter the reflectivity

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

#maxZe values
plt.plot_date(pd.to_datetime(time[:], unit='s'),maxZe, '+',
              markersize=2, markeredgewidth=1)
plt.xlim(pd.to_datetime(time[0], unit='s'),pd.to_datetime(time[-1], unit='s'))
plt.ylabel('max Ze value')
plt.xlabel('Time [UTC]')
plt.title('MRR OSUG-B ' +event_date+ ' \n Maximum reflectivity')
plt.grid()
plt.gcf().autofmt_xdate()
date_format = mdate.DateFormatter('%H:00')
plt.gca().xaxis.set_major_formatter(date_format)
plt.tight_layout()
plt.show()

#maxZe elevations
plt.plot_date(pd.to_datetime(time[:], unit='s'),hmaxZe, '+',
              markersize=2, markeredgewidth=1)
plt.xlim(pd.to_datetime(time[0], unit='s'),pd.to_datetime(time[-1], unit='s'))
plt.ylabel('max Ze elevation [m]')
plt.xlabel('Time [UTC]')
plt.title('MRR OSUG-B ' +event_date+ '\n Maximum reflectivity elevation')
plt.grid()
plt.gcf().autofmt_xdate()
date_format = mdate.DateFormatter('%H:%M')
plt.gca().xaxis.set_major_formatter(date_format)
plt.tight_layout()
plt.show()
#%% outliers filtering
hmaxZe = ma.masked_where(hmaxZe<400,hmaxZe) #remove the first bin close tothe ground

std_Ze = r
z_score = np.abs(stats.zscore(ma.filled(hmaxZe,fill_value=np.nan),
                              nan_policy='omit'))
print('remove all values above '+ str(hmaxZe.mean()+std_Ze*hmaxZe.std())+
      ' and below '+str(hmaxZe.mean()-std_Ze*hmaxZe.std()))
hmaxZe = ma.masked_where(z_score>std_Ze,hmaxZe) #set the mask as True where z_score>3

#max velocity gradient without outliers
plt.plot_date(pd.to_datetime(time[:], unit='s'),hmaxZe, '+',
              markersize=2, markeredgewidth=1)
plt.ylim(0, 3000)
plt.xlim(pd.to_datetime(time[0], unit='s'),pd.to_datetime(time[-1], unit='s'))
plt.ylabel('Height (m.a.i.l)')
plt.xlabel('Time [UTC]')
plt.title('MRR OSUG-B ' +event_date+ '\n Maximum reflectivity position')
plt.grid()
plt.gcf().autofmt_xdate()
date_format = mdate.DateFormatter('%H:%M')
plt.gca().xaxis.set_major_formatter(date_format)
plt.tight_layout()
plt.show()

#%%convert the masked array to pd.df and interpolate the NAN
hmaxZe_interpol=pd.DataFrame(hmaxZe).interpolate(method='linear', limit=time_interp, 
                                                   limit_area='inside')
hmaxZe_smooth = gaussian_filter1d(hmaxZe_interpol.values[:,0],sigma=sigma)
hmaxZe_smooth = ma.masked_values(hmaxZe_smooth, np.nan)
hmaxZe_smooth = ma.masked_array(data=hmaxZe_smooth.data, mask=np.isnan(hmaxZe_smooth.data)) #reconvert to masked array

plt.plot_date(pd.to_datetime(time[:].data, unit='s'),
              hmaxZe_smooth, '-', label=r'max(Ze)',
              markersize=3, markeredgewidth=1)
plt.ylim(0, 3000)
plt.xlim(pd.to_datetime(time[0], unit='s'),pd.to_datetime(time[-1], unit='s'))
plt.ylabel('Height (m.a.i.l)')
plt.xlabel('Time [UTC]')
plt.title('MRR OSUG-B ' +event_date+ '\n Maximum reflectivity elevation')
plt.legend(fontsize=10)
plt.grid(which='major')
plt.grid(which='minor', alpha=0.5)
plt.minorticks_on()
plt.gcf().autofmt_xdate()
date_format = mdate.DateFormatter('%H:00')
plt.gca().xaxis.set_major_formatter(date_format)
#plt.tight_layout()
plt.rc('xtick', labelsize=12) 
plt.show()

#%%compute gradZe
gradZe= np.gradient(Ze_filtered, axis=1)
grad2Ze= np.gradient(gradZe, axis=1)

#find the minimum gradZe
hML_lim= hmaxZe_smooth.mean() + 500 #do not consider reflectivities 500m above the ML height when searching for the ML top
mingradZe= ma.MaskedArray.min(ma.masked_where(height>hML_lim,gradZe[:,:]),axis=1) #we will use his mask to filter values 
#maxgradZe= ma.masked_where((maxgradZe<0.5)|(maxgradZe>2),maxgradZe) #only keep the strong gradients
imingradZe=ma.masked_array(ma.MaskedArray.argmin(gradZe[:,:],axis=1),fill_value=0,
                         mask=mingradZe.mask) #array of index of the maxima

hmingradZe=ma.masked_array(height[:,imingradZe.data][0],mask=imingradZe.mask)

plt.plot_date(pd.to_datetime(time[:], unit='s'),mingradZe, '+',
              markersize=2, markeredgewidth=1)
plt.xlim(pd.to_datetime(time[0], unit='s'),pd.to_datetime(time[-1], unit='s'))
plt.ylabel('min grad Ze')
plt.xlabel('Time [UTC]')
plt.title('MRR OSUG-B ' +event_date+ '\n Minimum reflectivity gradient value')
plt.grid()
plt.gcf().autofmt_xdate()
date_format = mdate.DateFormatter('%H:00')
plt.gca().xaxis.set_major_formatter(date_format)
plt.tight_layout()
plt.show()

plt.plot_date(pd.to_datetime(time[:], unit='s'),hmingradZe, '+',
              markersize=2, markeredgewidth=1)
plt.xlim(pd.to_datetime(time[0], unit='s'),pd.to_datetime(time[-1], unit='s'))
plt.ylabel('height')
plt.xlabel('Time [UTC]')
plt.title('MRR OSUG-B ' +event_date+ '\n Minimum reflectivity gradient position')
plt.grid()
plt.gcf().autofmt_xdate()
date_format = mdate.DateFormatter('%H:00')
plt.gca().xaxis.set_major_formatter(date_format)
plt.tight_layout()
plt.show()
#%%
hmingradZe = ma.masked_where(hmingradZe<400,hmingradZe)
hmingradZe = ma.masked_where(hmingradZe>(hmaxZe_smooth.mean()+500),hmingradZe) #restrict the search of MLtop to 500m above the MLheight
#hmingradZe = ma.masked_values(hmingradZe, hmaxZe_smooth) #masked the values of min grad where maxZe does not exist
std_gradZe = r
z_score = np.abs(stats.zscore(ma.filled(hmingradZe,fill_value=np.nan),
                              nan_policy='omit'))
print('remove all values above '+ str(hmingradZe.mean()+std_gradZe*hmingradZe.std())+
      ' and below '+str(hmingradZe.mean()-std_gradZe*hmingradZe.std()))

hmingradZe = ma.masked_where(z_score>std_gradZe,hmingradZe) #set the mask as True where z_score>3

#max velocity gradient without outliers
plt.plot_date(pd.to_datetime(time[:], unit='s'),hmingradZe, '+',
              markersize=2, markeredgewidth=1)
plt.ylim(0, 3000)
plt.xlim(pd.to_datetime(time[0], unit='s'),pd.to_datetime(time[-1], unit='s'))
plt.ylabel('Height (m.a.i.l)')
plt.xlabel('Time [UTC]')
plt.title('MRR OSUG-B ' +event_date+ '\n Minimum reflectivity gradient position')
plt.grid()
plt.gcf().autofmt_xdate()
date_format = mdate.DateFormatter('%H:00')
plt.gca().xaxis.set_major_formatter(date_format)
plt.tight_layout()
plt.show()
#%%
hmingradZe_interpol=pd.DataFrame(hmingradZe).interpolate(method='linear', limit=time_interp, 
                                                   limit_area='inside')
hmingradZe_smooth = gaussian_filter1d(hmingradZe_interpol.values[:,0],sigma=sigma)
hmingradZe_smooth = ma.masked_values(hmingradZe_smooth,  np.nan)
hmingradZe_smooth = ma.masked_array(data=hmingradZe_smooth.data, 
                                    mask=np.isnan(hmingradZe_smooth.data)) #reconvert to masked array

plt.plot_date(pd.to_datetime(time[:].data, unit='s'),
              hmaxZe_smooth, '-', label=r'max(Ze)',
              markersize=3, markeredgewidth=1)
plt.plot_date(pd.to_datetime(time[:].data, unit='s'),
              hmingradZe_smooth, '-', label=r'min(gradZe)',
              markersize=2, markeredgewidth=1)
plt.ylim(0, 3000)
plt.xlim(pd.to_datetime(time[0], unit='s'),pd.to_datetime(time[-1], unit='s'))
plt.ylabel('Height (m.a.i.l)')
plt.xlabel('Time [UTC]')
plt.title('MRR OSUG-B ' +event_date+ '\n Min reflectivity elevation')
plt.legend(fontsize=10)
plt.grid(which='major')
plt.grid(which='minor', alpha=0.5)
plt.minorticks_on()
plt.gcf().autofmt_xdate()
date_format = mdate.DateFormatter('%H:00')
plt.gca().xaxis.set_major_formatter(date_format)
#plt.tight_layout()
plt.rc('xtick', labelsize=12) 
plt.show()

#%% find the max gradZe (=== the bottom of the ML?)

#the max of gradient is searched for the values below the position of the peak of reflectivity only
gradZe_formax = gradZe
for i in range(0,len(time)):
    gradZe_formax[i,:] = ma.masked_where(np.arange(0,31)>=imaxZe[i],gradZe[i,:])
    
maxgradZe= ma.MaskedArray.max(gradZe_formax[:,:],axis=1) #we will use his mask to filter values 
#maxgradZe= ma.masked_where((maxgradZe<0.5)|(maxgradZe>2),maxgradZe) #only keep the strong gradients
imaxgradZe=ma.masked_array(ma.MaskedArray.argmax(gradZe_formax[:,:],axis=1),fill_value=0,
                         mask=maxgradZe.mask) #array of index of the maxima

hmaxgradZe=ma.masked_array(height[:,imaxgradZe.data][0],mask=imaxgradZe.mask)

plt.plot_date(pd.to_datetime(time[:], unit='s'),maxgradZe, '+',
              markersize=2, markeredgewidth=1)
plt.xlim(pd.to_datetime(time[0], unit='s'),pd.to_datetime(time[-1], unit='s'))
plt.ylabel('min grad Ze')
plt.xlabel('Time [UTC]')
plt.title('MRR OSUG-B ' +event_date+ '\n Minimum reflectivity gradient value')
plt.grid()
plt.gcf().autofmt_xdate()
date_format = mdate.DateFormatter('%H:00')
plt.gca().xaxis.set_major_formatter(date_format)
plt.tight_layout()
plt.show()

plt.plot_date(pd.to_datetime(time[:], unit='s'),hmaxgradZe, '+',
              markersize=2, markeredgewidth=1)
plt.xlim(pd.to_datetime(time[0], unit='s'),pd.to_datetime(time[-1], unit='s'))
plt.ylabel('height')
plt.xlabel('Time [UTC]')
plt.title('MRR OSUG-B ' +event_date+ '\n Minimum reflectivity gradient position')
plt.grid()
plt.gcf().autofmt_xdate()
date_format = mdate.DateFormatter('%H:00')
plt.gca().xaxis.set_major_formatter(date_format)
plt.tight_layout()
plt.show()

#%%
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

#max velocity gradient without outliers
plt.plot_date(pd.to_datetime(time[:], unit='s'),hmaxgradZe, '+',
              markersize=2, markeredgewidth=1)
plt.ylim(0, 3000)
plt.xlim(pd.to_datetime(time[0], unit='s'),pd.to_datetime(time[-1], unit='s'))
plt.ylabel('Height (m.a.i.l)')
plt.xlabel('Time [UTC]')
plt.title('MRR OSUG-B ' +event_date+ '\n Minimum reflectivity gradient position')
plt.grid()
plt.gcf().autofmt_xdate()
date_format = mdate.DateFormatter('%H:00')
plt.gca().xaxis.set_major_formatter(date_format)
plt.tight_layout()
plt.show()

#%%
hmaxgradZe_interpol=pd.DataFrame(hmaxgradZe).interpolate(method='linear', limit=time_interp, 
                                                   limit_area='inside')
#reject the values of the gradient elevation (set as NAN) if above the max reflectivity
hmaxgradZe_interpol[hmaxgradZe_interpol >= hmaxZe_interpol] = np.nan
hmaxgradZe_smooth = gaussian_filter1d(hmaxgradZe_interpol.values[:,0],sigma=sigma)
hmaxgradZe_smooth = ma.masked_values(hmaxgradZe_smooth, np.nan)
hmaxgradZe_smooth = ma.masked_array(data=hmaxgradZe_smooth.data, 
                                    mask=np.isnan(hmaxgradZe_smooth.data)) #reconvert to masked array

plt.plot_date(pd.to_datetime(time[:].data, unit='s'),
              hmaxZe_smooth, '-', label=r'max(Ze)',
              markersize=3, markeredgewidth=1)
plt.plot_date(pd.to_datetime(time[:].data, unit='s'),
              hmingradZe_smooth, '-', label=r'min(gradZe)',
              markersize=2, markeredgewidth=1)
plt.plot_date(pd.to_datetime(time[:].data, unit='s'),
              hmaxgradZe_smooth, '-', label=r'max(gradZe)',
              markersize=2, markeredgewidth=1)
plt.ylim(0, 3000)
plt.xlim(pd.to_datetime(time[0], unit='s'),pd.to_datetime(time[-1], unit='s'))
plt.ylabel('Height (m.a.i.l)')
plt.xlabel('Time [UTC]')
plt.title('MRR OSUG-B ' +event_date + '\n ML elevation from reflectivity')
plt.legend(fontsize=10)
plt.grid(which='major')
plt.grid(which='minor', alpha=0.5)
plt.minorticks_on()
plt.gcf().autofmt_xdate()
date_format = mdate.DateFormatter('%d/%m %H:00')
plt.gca().xaxis.set_major_formatter(date_format)
#plt.tight_layout()
plt.rc('xtick', labelsize=12) 
plt.show()

#%%plot of profile at sepecific time (e.g. sat_time)
id_sat_time = np.argwhere(pd.to_datetime(time[:].data, unit='s')==sat_time)[0][0]
Ze_avg = Ze_filtered[id_sat_time-5:id_sat_time+5,:].mean(axis=0)

fig = plt.figure(facecolor='white')
ax1 = fig.add_subplot(111)
ax2 = ax1.twiny()
ax1.set_xlim(0,35)
ax2.set_xlim(-4,6)
p1=ax1.plot(Ze_avg, height[id_sat_time,:], '-+',c='b', label='Ze')
p2=ax2.plot(gradZe[id_sat_time-5:id_sat_time+5,:].mean(axis=0), height[id_sat_time,:], '-+',c='r', label=r'$\partial_{z}{Ze}$')
ax2.set_xlabel(r"$10^{-2}.dBZ.m^{-1}$", color='r')
ax2.tick_params(axis='x', labelcolor='r')
ax1.set_xlabel("dBZ", color='b')
ax1.tick_params(axis='x', labelcolor='b')
ax1.set_ylabel("height [m.a.g.l]")
h1=ax1.axhline(hmaxZe[id_sat_time], ls='--', alpha=0.7, label='max(Ze)')
h2=ax1.axhline(hmaxgradZe[id_sat_time], ls='--',color='g', alpha=0.7, label=r'max($\partial_{z}{Ze}$)')
h3=ax1.axhline(hmingradZe[id_sat_time], ls='--',color='orange', alpha=0.7, label=r'min($\partial_{z}{Ze}$)')
plt.legend(handles=p1+p2+[h1]+[h2]+[h3])
#labs = [l.get_label() for l in (p1+p2+[h1]+[h2]+[h3])]
#ax1.legend(p1+p2+[h1]+[h2]+[h3],labs,fontsize=10, loc=0)
fig.tight_layout()
plt.show()

#%%### ML detection from Velocity W

#filter the velocity
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

plt.plot_date(pd.to_datetime(time[:], unit='s'),hmaxgrad, '+',
              markersize=2, markeredgewidth=1)
plt.xlim(pd.to_datetime(time[0], unit='s'),pd.to_datetime(time[-1], unit='s'))
plt.ylabel('max grad W')
plt.xlabel('Time [UTC]')
plt.title('MRR OSUG-B 2019-11-14 \n Maximum velocity gradient value')
plt.grid()
plt.gcf().autofmt_xdate()
date_format = mdate.DateFormatter('%H:00')
plt.gca().xaxis.set_major_formatter(date_format)
plt.tight_layout()
plt.show()
#%%
std_gradW = r
z_score = np.abs(stats.zscore(ma.filled(hmaxgrad,fill_value=np.nan),
                              nan_policy='omit'))
print('remove all values above '+ str(hmaxgrad.mean()+std_gradW*hmaxgrad.std())+
      ' and below '+str(hmaxgrad.mean()-std_gradW*hmaxgrad.std()))

hmaxgrad = ma.masked_where(z_score>std_gradW,hmaxgrad) #set the mask as True where z_score>3

#max velocity gradient without outliers
plt.plot_date(pd.to_datetime(time[:], unit='s'),hmaxgrad, '+',
              markersize=2, markeredgewidth=1)
plt.ylim(0, 3000)
plt.xlim(pd.to_datetime(time[0], unit='s'),pd.to_datetime(time[-1], unit='s'))
plt.ylabel('Height (m.a.i.l)')
plt.xlabel('Time [UTC]')
plt.title('MRR OSUG-B 2019-11-14 \n Maximum velocity gradient position')
plt.grid()
plt.gcf().autofmt_xdate()
date_format = mdate.DateFormatter('%H:00')
plt.gca().xaxis.set_major_formatter(date_format)
plt.tight_layout()
plt.show()
#%%
h_interpol=pd.DataFrame(hmaxgrad).interpolate(method='linear', limit=time_interp, 
                                                   limit_area='inside')
h_smooth = gaussian_filter1d(h_interpol.values[:,0],sigma=sigma)
h_smooth = ma.masked_values(h_smooth,  np.nan)
h_smooth = ma.masked_array(data=h_smooth.data, mask=np.isnan(h_smooth.data)) #reconvert to masked array

plt.plot_date(pd.to_datetime(time[:].data, unit='s'),
              h_smooth, '-', label=r'max(gradW)',
              markersize=1, markeredgewidth=1)
plt.ylim(0, 3000)
plt.xlim(pd.to_datetime(time[0], unit='s'),pd.to_datetime(time[-1], unit='s'))
plt.ylabel('Height (m.a.i.l)')
plt.xlabel('Time [UTC]')
plt.title('MRR OSUG-B 2019-11-14 \n Maximum velocity gradient position')
plt.grid()
plt.gcf().autofmt_xdate()
date_format = mdate.DateFormatter('%H:00')
plt.gca().xaxis.set_major_formatter(date_format)
plt.tight_layout()
plt.show()

#%%## compute grad2W
grad2W= -np.gradient(gradW, axis=1)

## find the max of grad2W
hML_lim= h_smooth.mean() + 500 #do not consider velocities 500m above the ML height when searching for the ML top
maxgrad2W= ma.MaskedArray.max(ma.masked_where(height>hML_lim,grad2W),axis=1) #we will use his mask to filter values 

plt.plot_date(pd.to_datetime(time[:], unit='s'),maxgrad2W, '+',
              markersize=2, markeredgewidth=1)
plt.xlim(pd.to_datetime(time[0], unit='s'),pd.to_datetime(time[-1], unit='s'))
plt.ylabel('maxgrad2W')
plt.xlabel('Time [UTC]')
plt.title('MRR OSUG-B 2019-11-14 \n Maximum second velocity gradient value')
plt.grid()
plt.gcf().autofmt_xdate()
date_format = mdate.DateFormatter('%H:00')
plt.gca().xaxis.set_major_formatter(date_format)
plt.tight_layout()
plt.show()

maxgrad2W= ma.masked_where((maxgrad2W<0.25)|(maxgrad2W>3),maxgrad2W) #only keep the strong gradients
imaxgrad2W=ma.masked_array(ma.MaskedArray.argmax(
    ma.masked_where(height>hML_lim,grad2W),axis=1),
    fill_value=0,
                         mask=maxgrad2W.mask) #array of index of the maxima
hmaxgrad2W=ma.masked_array(height[:,imaxgrad2W.data][0],mask=imaxgrad2W.mask)
plt.plot_date(pd.to_datetime(time[:].data, unit='s'),
              hmaxgrad2W, '+',
              markersize=2, markeredgewidth=1)
plt.ylim(0, 3000)
plt.xlim(pd.to_datetime(time[0], unit='s'),pd.to_datetime(time[-1], unit='s'))
plt.ylabel('Height (m.a.i.l)')
plt.xlabel('Time [UTC]')
plt.title('MRR OSUG-B '+event_date+'\n Maximum second velocity gradient position')
plt.grid()
plt.gcf().autofmt_xdate()
date_format = mdate.DateFormatter('%H:00')
plt.gca().xaxis.set_major_formatter(date_format)
plt.tight_layout()
plt.show()

hmaxgrad2W = ma.masked_values(hmaxgrad2W, h_smooth) #masked the values of min grad where maxZe does not exist
#%%filter the maximum values of the gradient which are outside mean+/- X std
std_grad2W = r
z_score2 = np.abs(stats.zscore(ma.filled(hmaxgrad2W,fill_value=np.nan),
                              nan_policy='omit'))
print('remove all values above '+ str(hmaxgrad2W.mean()+std_grad2W*hmaxgrad2W.std())+
      ' and below '+str(hmaxgrad2W.mean()-std_grad2W*hmaxgrad2W.std()))

hmaxgrad2W = ma.masked_where(z_score2 > std_grad2W, hmaxgrad2W) #set the mask as True where z_score>3

#reject the values of the second derivative where the values of the first derivative where rejected,
#i.e. the second gradient maximum can exist only if the max of first derivative exists
#hmaxgrad2W = ma.masked_where(maxgradW.mask==True, hmaxgrad2W)

plt.plot_date(pd.to_datetime(time[:].data, unit='s'),
              ma.masked_where(maxgradW.mask==True, hmaxgrad2W), '+',
              markersize=2, markeredgewidth=1)
plt.ylim(0, 3000)
plt.xlim(pd.to_datetime(time[0], unit='s'),pd.to_datetime(time[-1], unit='s'))
plt.ylabel('Height (m.a.i.l)')
plt.xlabel('Time [UTC]')
plt.title('MRR OSUG-B '+event_date+'\n Maximum second velocity gradient position')
plt.grid()
plt.gcf().autofmt_xdate()
date_format = mdate.DateFormatter('%H:00')
plt.gca().xaxis.set_major_formatter(date_format)
plt.tight_layout()
plt.show()

#%%convert the masked array to pd.df and interpolate the NAN
hmaxgrad2W_interpol=pd.DataFrame(hmaxgrad2W).interpolate(method='linear', limit=time_interp, 
                                                   limit_area='inside')
hmaxgrad2W_smooth = gaussian_filter1d(hmaxgrad2W_interpol.values[:,0],sigma=sigma)
hmaxgrad2W_smooth = ma.masked_values(hmaxgrad2W_smooth,  np.nan)
hmaxgrad2W_smooth = ma.masked_array(data=hmaxgrad2W_smooth.data, mask=np.isnan(hmaxgrad2W_smooth.data)) #reconvert to masked array
hmaxgrad2W_smooth = ma.masked_where(hmaxgrad2W_smooth<h_smooth, hmaxgrad2W_smooth) #top of ML can't be lower than the height of ML

plt.plot_date(pd.to_datetime(time[:].data, unit='s'),
              h_smooth, '-',
              markersize=2, markeredgewidth=1)

plt.plot_date(pd.to_datetime(time[:].data, unit='s'),
              hmaxgrad2W_smooth, '-',
              markersize=1, markeredgewidth=1)
plt.ylim(0, 3000)
plt.xlim(pd.to_datetime(time[0], unit='s'),pd.to_datetime(time[-1], unit='s'))
plt.ylabel('Height (m.a.i.l)')
plt.xlabel('Time [UTC]')
plt.title('MRR OSUG-B '+event_date+'\n Maximum second velocity gradient position')
plt.grid(which='major')
plt.grid(which='minor', alpha=0.5)
plt.minorticks_on()
plt.gcf().autofmt_xdate()
date_format = mdate.DateFormatter('%H:00')
plt.gca().xaxis.set_major_formatter(date_format)
plt.tight_layout()
plt.show()

#%%# find the min of grad2
mingrad2W= ma.MaskedArray.min(ma.masked_where(height>hML_lim,grad2W),axis=1) #we will use his mask to filter values 

plt.plot_date(pd.to_datetime(time[:], unit='s'),mingrad2W, '+',
              markersize=2, markeredgewidth=1)
plt.xlim(pd.to_datetime(time[0], unit='s'),pd.to_datetime(time[-1], unit='s'))
plt.ylabel('mingrad2W')
plt.xlabel('Time [UTC]')
plt.title('MRR OSUG-B '+event_date+'\n Minimum second velocity gradient value')
plt.grid()
plt.gcf().autofmt_xdate()
date_format = mdate.DateFormatter('%H:00')
plt.gca().xaxis.set_major_formatter(date_format)
plt.tight_layout()
plt.show()

mingrad2W= ma.masked_where((np.abs(mingrad2W)<0.2) |
                           (np.abs(mingrad2W)>3),mingrad2W) #only keep the strong gradients
imingrad2W=ma.masked_array(
    ma.MaskedArray.argmin(ma.masked_where(height>hML_lim,grad2W),axis=1),
    fill_value=0, mask=mingrad2W.mask) #array of index of the maxima
hmingrad2W= ma.masked_array(height[:,imingrad2W.data][0],mask=imingrad2W.mask)
hmingrad2W= ma.masked_where(hmingrad2W<=500, hmingrad2W) #minimum can't be less or equal to 500m (because it is the boundary) 

plt.plot_date(pd.to_datetime(time[:].data, unit='s'),
              hmingrad2W, '+',
              markersize=2, markeredgewidth=1)
plt.ylim(0, 3000)
plt.xlim(pd.to_datetime(time[0], unit='s'),pd.to_datetime(time[-1], unit='s'))
plt.ylabel('Height (m.a.i.l)')
plt.xlabel('Time [UTC]')
plt.title('MRR OSUG-B '+event_date+'\n Min 2nd velocity gradient position')
plt.grid()
plt.gcf().autofmt_xdate()
date_format = mdate.DateFormatter('%H:00')
plt.gca().xaxis.set_major_formatter(date_format)
plt.tight_layout()
plt.show()

hmingrad2W = ma.masked_values(hmingrad2W, h_smooth) #masked the values of min grad2 where maxgrad does not exist
#%%filter the maximum values of the gradient which are outside mean+/-3std
std_grad2W = r
z_score3 = np.abs(stats.zscore(ma.filled(hmingrad2W,fill_value=np.nan),
                              nan_policy='omit'))
print('remove all values above '+ str(hmingrad2W.mean()+std_grad2W*hmingrad2W.std())+
      ' and below '+str(hmingrad2W.mean()-std_grad2W*hmingrad2W.std()))

hmingrad2W = ma.masked_where(z_score3 > std_grad2W, hmingrad2W) #set the mask as True where z_score>3

#reject the values of the second derivative where the values of the first derivative where rejected,
#i.e. the second gradient maximum can exist only if the max of first derivative exists
#hmingrad2W = ma.masked_where(maxgradW.mask==True, hmingrad2W)

plt.plot_date(pd.to_datetime(time[:].data, unit='s'),
              hmingrad2W, '+',
              markersize=2, markeredgewidth=1)
plt.ylim(0, 3000)
plt.xlim(pd.to_datetime(time[0], unit='s'),pd.to_datetime(time[-1], unit='s'))
plt.ylabel('Height (m.a.i.l)')
plt.xlabel('Time [UTC]')
plt.title('MRR OSUG-B '+event_date+'\n Min 2nd velocity gradient position')
plt.grid()
plt.gcf().autofmt_xdate()
date_format = mdate.DateFormatter('%H:00')
plt.gca().xaxis.set_major_formatter(date_format)
plt.tight_layout()
plt.show()

#remove the values wich have an elevation above are equal to those of the first derivative
hmingrad2W = ma.masked_where(hmingrad2W >= h_smooth, hmingrad2W)
#hmingrad2W = ma.masked_where(h_smooth.mask, hmingrad2W)

#%%convert the masked array to pd.df and interpolate the NAN
hmingrad2W_interpol=pd.DataFrame(hmingrad2W).interpolate(method='linear', limit=time_interp, 
                                                   limit_area='inside')
hmingrad2W_smooth = gaussian_filter1d(hmingrad2W_interpol.values[:,0],sigma=sigma)
hmingrad2W_smooth = ma.masked_values(hmingrad2W_smooth, np.nan)
hmingrad2W_smooth = ma.masked_array(data=hmingrad2W_smooth.data, mask=np.isnan(hmingrad2W_smooth.data)) #reconvert to masked array#reconvert to masked array
hmingrad2W_smooth = ma.masked_where(hmingrad2W_smooth>h_smooth, hmingrad2W_smooth) #ML bottom cant be above ML height

plt.plot_date(pd.to_datetime(time[:].data, unit='s'),
              h_smooth, '-', label=r'max($\nabla W$)',
              markersize=3, markeredgewidth=1)
plt.plot_date(pd.to_datetime(time[:].data, unit='s'),
              hmaxgrad2W_smooth, '-', label=r'max($\nabla^{2} W$)',
              markersize=2, markeredgewidth=1)
plt.plot_date(pd.to_datetime(time[:].data, unit='s'),
              hmingrad2W_smooth, '-', label=r'min($\nabla^{2} W$)',
              markersize=2, markeredgewidth=1)
plt.ylim(0, 3000)
plt.xlim(pd.to_datetime(time[0], unit='s'),pd.to_datetime(time[-1], unit='s'))
plt.ylabel('Height (m.a.g.l)')
plt.xlabel('Time [UTC]')
plt.title('MRR OSUG-B '+event_date+'\n $\partial_{z} W}$ and $\partial_{z}^{2} W}$ extrema positions')
plt.legend(fontsize=10)
plt.grid(which='major')
plt.grid(which='minor', alpha=0.5)
plt.minorticks_on()
plt.gcf().autofmt_xdate()
date_format = mdate.DateFormatter('%H:00')
plt.gca().xaxis.set_major_formatter(date_format)
#plt.tight_layout()
plt.rc('xtick', labelsize=12) 
plt.show()

#%%plot of profile at specific time
id_sat_time = np.argwhere(pd.to_datetime(time[:].data, unit='s')==sat_time)[0][0]

fig = plt.figure(facecolor='white')
ax1 = fig.add_subplot(111)
ax2 = ax1.twiny()
ax3 = ax1.twiny()
ax3.spines['top'].set_position(('outward', 30))
ax3.set_xlim(-1,2)
ax2.set_xlim(-0.3,2.7)
ax1.set_xlim(0,6)
ax1.set_ylim(0,3000)
p1=ax1.plot(W_filt[id_sat_time-5:id_sat_time+5,:].mean(axis=0), height[id_sat_time,:], '-+',c='b', label='W')
p2=ax2.plot(gradW[id_sat_time,:], height[id_sat_time,:], '-+',c='r', label=r'$\partial_{z}{W}$')
p3=ax3.plot(grad2W[id_sat_time,:], height[id_sat_time,:], '-+',c='purple', label=r'$\partial_{z}^2{W}$')
ax2.set_xlabel(r"$10^{-2}.s^{-1}$", color='r')
ax2.tick_params(axis='x', labelcolor='r')
ax1.set_xlabel(r"$m.s^{-1}$", color='b')
ax1.tick_params(axis='x', labelcolor='b')
ax3.set_xlabel(r"$10^{-4}.m^{-1}.s^{-1}$", color='purple')
ax3.tick_params(axis='x', labelcolor='purple')
ax1.set_ylabel("height [m.a.g.l]")
h1=ax1.axhline(hmaxgrad[id_sat_time], ls='--', alpha=0.6, label=r'max($\partial_{z}(W)$')
h2=ax1.axhline(hmaxgrad2W[id_sat_time], ls='--',color='orange', alpha=0.6, label=r'max($\partial_{z}^2{W}$)')
h3=ax1.axhline(hmingrad2W[id_sat_time], ls='--',color='g', alpha=0.6, label=r'min($\partial_{z}^2{W}$)')
plt.legend(handles=p1+p2+p3+[h1]+[h2]+[h3], ncol=2)
#plt.plot(grad2Ze[1499,:], height[1499,:], '-+', label='grad2Ze')
#labs = [l.get_label() for l in (p1+p2+[h1]+[h2]+[h3])]
#ax1.legend(p1+p2+[h1]+[h2]+[h3],labs,fontsize=10, loc=0)
fig.tight_layout()
plt.show()


#%%plot reflectivity and velocity ML in m.a.s.l sat_time

plt.figure(facecolor='white', figsize=(8,5))
plt.plot_date(pd.to_datetime(time[:].data, unit='s'),
              h_smooth + alt_station, '-', label=r'$max(\partial_z{W})$',
              markersize=1, markeredgewidth=1, alpha=0.9)
plt.plot_date(pd.to_datetime(time[:].data, unit='s'),
              hmaxgrad2W_smooth + alt_station, '-', label=r'$max(\partial_z^2{W})$',
              markersize=1, markeredgewidth=1, alpha=0.9)
plt.plot_date(pd.to_datetime(time[:].data, unit='s'),
              hmingrad2W_smooth + alt_station, '-', label=r'$min(\partial_z{W})$',
              markersize=1, markeredgewidth=1, alpha=0.9)
plt.plot_date(pd.to_datetime(time[:].data, unit='s'),
              hmaxZe_smooth + alt_station, '--', label=r'$max(Ze)$', color='tab:blue',
              markersize=1, markeredgewidth=1, alpha=0.9)
plt.plot_date(pd.to_datetime(time[:].data, unit='s'),
              hmingradZe_smooth + alt_station, '--', label=r'$min(\partial_z{Ze})$', color='tab:orange',
              markersize=1, markeredgewidth=1, alpha=0.9)
plt.plot_date(pd.to_datetime(time[:].data, unit='s'),
              hmaxgradZe_smooth + alt_station, '--', label=r'$max(\partial_z{Ze})$', color='tab:green',
              markersize=1, markeredgewidth=1, alpha=0.9)
plt.ylim(0, 3500)
plt.xlim(pd.to_datetime(time[0], unit='s'),pd.to_datetime(time[-1], unit='s'))
plt.ylabel('Elevation [m.a.s.l]')
plt.xlabel('Time [UTC]')
#plt.title('MRR OSUG-B '+event_date+'\n ML boundaries elevation')
plt.legend(fontsize=10, ncol=2)
plt.grid(which='minor', alpha=0.1)
plt.grid(which='major', alpha=0.3)
plt.minorticks_on()
plt.gcf().autofmt_xdate()
date_format = mdate.DateFormatter('%d%b \n %H:%M')
plt.gca().xaxis.set_major_formatter(date_format)
plt.tight_layout()
plt.rc('xtick', labelsize=10) 
plt.show()

#%% Plot ML boundaries with OSUG, SMU and Chamrousse elevation and time satellite 

plt.figure(facecolor='white', figsize=(8,5))
plt.plot_date(pd.to_datetime(time[:].data, unit='s'),
              h_smooth + alt_station, '-', label=r'$max(\partial_z{W})$',
              markersize=1, markeredgewidth=1, alpha=0.9)
plt.plot_date(pd.to_datetime(time[:].data, unit='s'),
              hmaxgrad2W_smooth + alt_station, '-', label=r'$max(\partial_z^2{W})$',
              markersize=1, markeredgewidth=1, alpha=0.9)
plt.plot_date(pd.to_datetime(time[:].data, unit='s'),
              hmingrad2W_smooth + alt_station, '-', label=r'$min(\partial_z{W})$',
              markersize=1, markeredgewidth=1, alpha=0.9)
plt.plot_date(pd.to_datetime(time[:].data, unit='s'),
              hmaxZe_smooth + alt_station, '--', label=r'$max(Ze)$', color='tab:blue',
              markersize=1, markeredgewidth=1, alpha=0.9)
plt.plot_date(pd.to_datetime(time[:].data, unit='s'),
              hmingradZe_smooth + alt_station, '--', label=r'$min(\partial_z{Ze})$', color='tab:orange',
              markersize=1, markeredgewidth=1, alpha=0.9)
plt.plot_date(pd.to_datetime(time[:].data, unit='s'),
              hmaxgradZe_smooth + alt_station, '--', label=r'$max(\partial_z{Ze})$', color='tab:green',
              markersize=1, markeredgewidth=1, alpha=0.9)
# plt.axhline(y= Stations.loc['OSUG_B']['Elev'], alpha= 0.6, linestyle='dashed', color='k')
# plt.text(x=pd.to_datetime(time[0], unit='s'),y=Stations.loc['OSUG_B']['Elev'],
#         s='OSUG-B', verticalalignment='bottom', horizontalalignment='left', color='k')
# plt.axhline(y= Stations.loc['Chamrousse']['Elev'], alpha= 0.6, linestyle='dashed', color='k')
# plt.text(x=pd.to_datetime(time[0], unit='s'),y=Stations.loc['Chamrousse']['Elev'],
#         s='Chamrousse', verticalalignment='bottom', horizontalalignment='left')
# plt.axhline(y= Stations.loc['SMU']['Elev'], alpha= 0.6, linestyle='dashed', color='k')
# plt.text(x=pd.to_datetime(time[0], unit='s'),y=Stations.loc['SMU']['Elev'],
#         s='SMU', verticalalignment='bottom', horizontalalignment='left')
plt.axvline(x= datetime.strptime(sat_time, '%Y-%m-%d %H:%M:%S'),  alpha= 0.6, linestyle='dashed', color='k')
plt.ylim(0, 3000)
plt.xlim(pd.to_datetime(time[0], unit='s'),pd.to_datetime(time[-1], unit='s'))
plt.ylabel('Elevation (m.a.s.l)')
plt.xlabel('Time [UTC]')
#plt.title('MRR OSUG-B '+event_date+'\n ML boundaries elevation')
plt.legend(fontsize=10, ncol=2)
plt.grid(which='minor', alpha=0.1)
plt.grid(which='major', alpha=0.3)
plt.minorticks_on()
plt.gcf().autofmt_xdate()
date_format = mdate.DateFormatter('%d%b \n %H:%M')
plt.gca().xaxis.set_major_formatter(date_format)
plt.tight_layout()
plt.rc('xtick', labelsize=10) 
plt.show()

#%% return ML elevations at the specific time
sat_time
Avg = 10 #min
id_sat_time = np.where(pd.to_datetime(time[:].data, unit='s') == sat_time)[0][0]
h_smooth[id_sat_time]
BB_MRR = pd.Series(dtype='float')
BB_MRR['heightBB_Ze'] = hmaxZe_smooth[id_sat_time-Avg:id_sat_time+Avg].mean() + alt_station
BB_MRR['topBB_Ze'] = hmingradZe_smooth[id_sat_time-Avg:id_sat_time+Avg].mean() + alt_station
BB_MRR['botBB_Ze'] = hmaxgradZe_smooth[id_sat_time-Avg:id_sat_time+Avg].mean() + alt_station
BB_MRR['heightBB_W'] = h_smooth[id_sat_time-Avg:id_sat_time+Avg].mean() + alt_station
BB_MRR['topBB_W'] = hmaxgrad2W_smooth[id_sat_time-Avg:id_sat_time+Avg].mean() + alt_station
BB_MRR['botBB_W'] = hmingrad2W_smooth[id_sat_time-Avg:id_sat_time+Avg].mean() + alt_station

print(BB_MRR)
BB_MRR.astype('float')

#%%
MRRvsGPM = pd.read_csv('C:/Users/rebouda/Documents/04-WorkResults/MRRvsGPM_2016-2022_less5pxl.csv',
                       index_col=0, sep = ';', parse_dates=True)
MRRvsGPM.index
sat_time
DPR_BB = MRRvsGPM.loc['2022-04-07 17:02:28']
DPR_BB = MRRvsGPM.loc['2021-10-04 23:04:51']
DPR_BB = MRRvsGPM.loc['2020-03-12 22:04:48']
DPR_BB
#%%
plt.figure(facecolor='white', figsize=(8,5))
plt.plot_date(pd.to_datetime(time[:].data, unit='s'),
              h_smooth + alt_station, '-', label=r'$max(\partial_z{W})$',
              markersize=1, markeredgewidth=1, alpha=0.9)
plt.plot_date(pd.to_datetime(time[:].data, unit='s'),
              hmaxgrad2W_smooth + alt_station, '-', label=r'$max(\partial_z^2{W})$',
              markersize=1, markeredgewidth=1, alpha=0.9)
plt.plot_date(pd.to_datetime(time[:].data, unit='s'),
              hmingrad2W_smooth + alt_station, '-', label=r'$min(\partial_z{W})$',
              markersize=1, markeredgewidth=1, alpha=0.9)
plt.plot_date(pd.to_datetime(time[:].data, unit='s'),
              hmaxZe_smooth + alt_station, '--', label=r'$max(Ze)$', color='tab:blue',
              markersize=1, markeredgewidth=1, alpha=0.9)
plt.plot_date(pd.to_datetime(time[:].data, unit='s'),
              hmingradZe_smooth + alt_station, '--', label=r'$min(\partial_z{Ze})$', color='tab:orange',
              markersize=1, markeredgewidth=1, alpha=0.9)
plt.plot_date(pd.to_datetime(time[:].data, unit='s'),
              hmaxgradZe_smooth + alt_station, '--', label=r'$max(\partial_z{Ze})$', color='tab:green',
              markersize=1, markeredgewidth=1, alpha=0.9)
plt.ylim(1500, 3000)
plt.xlim(pd.to_datetime(time[0], unit='s'),pd.to_datetime(time[-1], unit='s'))
plt.axvline(x= datetime.strptime(sat_time, '%Y-%m-%d %H:%M:%S'),  alpha= 0.3, linestyle='dashed', color='k')

plt.errorbar(x=datetime.strptime(sat_time, '%Y-%m-%d %H:%M:%S'), 
             y=DPR_BB.heightBB_med, yerr=82, fmt='x', capsize=5, color='tab:blue',
             elinewidth=3, ms=10, capthick=3)
plt.errorbar(x=datetime.strptime(sat_time, '%Y-%m-%d %H:%M:%S'), 
             y=DPR_BB.topBB_med, yerr=156, fmt='x', capsize=5, color='tab:orange',
             elinewidth=3, ms=10, capthick=3)
plt.errorbar(x=datetime.strptime(sat_time, '%Y-%m-%d %H:%M:%S'), 
             y=DPR_BB.botBB_med, yerr=122, fmt='x', capsize=5, color='tab:green',
             elinewidth=3, ms=10, capthick=3)

a=plt.errorbar(x=0,y=0, yerr=0, fmt='x', capsize=3, color='k', 
               label='DPR: median '+r'$\pm\sigma$')

plt.ylabel('Height [m.a.s.l]')
plt.xlabel('Time [UTC]')
#plt.title('MRR OSUG-B '+event_date+'\n ML boundaries elevation')
plt.legend(fontsize=10, ncol=2)
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

legend_elements = [Line2D([0], [0], color='k', lw=1, label='MRR: W_method'),
                   Line2D([0], [0], color='k', lw=1, linestyle='dashed', label='MRR: Ze_method'),
                   #Line2D([0], [0], marker='x', color='k', solid_joinstyle='miter', label='DPR: median '+r'$\pm\sigma$',markerfacecolor='g', markersize=5),
                   a,
                   Patch(facecolor='tab:orange', edgecolor='tab:orange',label='ML_top'),
                   Patch(facecolor='tab:blue', edgecolor='tab:blue',label='ML_height'),
                   Patch(facecolor='tab:green', edgecolor='tab:green',label='ML_bottom')
                   ]
plt.legend(handles=legend_elements, ncol=2)

plt.grid(which='minor', alpha=0.1)
plt.grid(which='major', alpha=0.3)
plt.minorticks_on()
plt.gcf().autofmt_xdate()
date_format = mdate.DateFormatter('%d%b\n%H:%M')
plt.gca().xaxis.set_major_formatter(date_format)
plt.tight_layout()
plt.rc('xtick', labelsize=10) 
plt.show()
#plt.savefig(path_save+'202003122205.pdf', format='pdf')

#%% Plot ML boundaries with OSUG, SMU and Chamrousse elevation and time satellite 

xlim = [datetime.strptime(start_time, '%Y-%m-%d %H:%M:%S'),datetime.strptime(end_time, '%Y-%m-%d %H:%M:%S')]
fig, ax1 = plt.subplots(facecolor='white', figsize=(10,5))

#plt.figure(facecolor='white', figsize=(8,5))

ax1.plot_date(pd.to_datetime(time[:].data, unit='s'),
              h_smooth + alt_station, '-', label=r'$max(\partial_z{W})$',
              markersize=1, markeredgewidth=1, alpha=0.9)
ax1.plot_date(pd.to_datetime(time[:].data, unit='s'),
              hmaxgrad2W_smooth + alt_station, '-', label=r'$max(\partial_z^2{W})$',
              markersize=1, markeredgewidth=1, alpha=0.9)
ax1.plot_date(pd.to_datetime(time[:].data, unit='s'),
              hmingrad2W_smooth + alt_station, '-', label=r'$min(\partial_z{W})$',
              markersize=1, markeredgewidth=1, alpha=0.9)
ax1.plot_date(pd.to_datetime(time[:].data, unit='s'),
              hmaxZe_smooth + alt_station, '--', label=r'$max(Ze)$', color='tab:blue',
              markersize=1, markeredgewidth=1, alpha=0.9)
ax1.plot_date(pd.to_datetime(time[:].data, unit='s'),
              hmingradZe_smooth + alt_station, '--', label=r'$min(\partial_z{Ze})$', color='tab:orange',
              markersize=1, markeredgewidth=1, alpha=0.9)
ax1.plot_date(pd.to_datetime(time[:].data, unit='s'),
              hmaxgradZe_smooth + alt_station, '--', label=r'$max(\partial_z{Ze})$', color='tab:green',
              markersize=1, markeredgewidth=1, alpha=0.9)
#ax1.axhline(y= Stations.loc['OSUG_B']['Elev'], alpha= 0.6, linestyle='dashed', color='k')
ax1.text(x=xlim[0],y=Stations.loc['OSUG_B']['Elev'],
        s=' OSUG-B', verticalalignment='bottom', horizontalalignment='left', color='k',zorder=6)
#ax1.axhline(y= Stations.loc['Chamrousse']['Elev'], alpha= 0.6, linestyle='dashed', color='k')
ax1.text(x=xlim[0],y=Stations.loc['Chamrousse']['Elev'],
        s=' Chamrousse', verticalalignment='bottom', horizontalalignment='left',zorder=6)
#ax1.axhline(y= Stations.loc['SMU']['Elev'], alpha= 0.6, linestyle='dashed', color='k')
ax1.text(x=xlim[0],y=Stations.loc['SMU']['Elev'],
        s=' SMU', verticalalignment='bottom', horizontalalignment='left',zorder=6)
ax1.axvline(x= datetime.strptime(sat_time, '%Y-%m-%d %H:%M:%S'),  alpha= 0.6, linestyle='dotted', color='k', zorder=5.5)
ax1.set_ylim(0, 2700)
#ax1.set_xlim(pd.to_datetime(time[0], unit='s'),pd.to_datetime(time[-1], unit='s'))
ax1.set_xlim(xlim[0],xlim[1])
plt.ylabel('Height [m.a.s.l]')
plt.xlabel('Time [UTC]')
#plt.title('MRR OSUG-B '+event_date+'\n ML boundaries elevation')
#ax1.legend(fontsize=10, ncol=2)
ax1.grid(which='minor', alpha=0.1)
ax1.grid(which='major', alpha=0.3)
ax1.minorticks_on()

legend_elements = [Line2D([-1], [0], linewidth=0, label='MRR method:'),
                    Line2D([0], [0], color='k', lw=1, label='Doppler velocity'),
                   Line2D([0], [0], color='k', lw=1, linestyle='dashed', label='Reflectivity'),
                   Patch(facecolor='tab:orange', edgecolor='tab:orange',label='ML_top'),
                   Patch(facecolor='tab:blue', edgecolor='tab:blue',label='ML_height'),
                   Patch(facecolor='tab:green', edgecolor='tab:green',label='ML_bottom')
                   ]
ax1.legend(handles=legend_elements, ncol=2, 
           bbox_to_anchor=(0.4, 0.985), borderaxespad=0)

# plt.bar(df_phase_mix.index, df_phase_mix['Solid'],width=0.001, label='Solid', color='light blue')
# plt.bar(df_phase_mix.index, df_phase_mix['Liquid'],width=0.001, bottom=df_phase_mix['Solid'],
#        label='Liquid', color='light red')
# plt.bar(df_phase_mix.index, df_phase_mix['Mixture'],width=0.001, bottom=df_phase_mix['Liquid'],
#        label='Mixture', color='light green')
# plt.legend()

ax2 = ax1.inset_axes([ax1.get_xlim()[0], Stations.loc['OSUG_B']['Elev'], ax1.get_xlim()[1]-ax1.get_xlim()[0], 400],
                     transform=ax1.transData)
ax2.set_facecolor('grey')
ax2.patch.set_alpha(0.2)
ax2.set_xlim(xlim[0],xlim[1])
ax2.set_ylim(0,1000)
#ax2.plot(pd.to_datetime(time[:].data, unit='s'),h_smooth)
ax2.bar(df_phase_mix_osug2101.index, df_phase_mix_osug2101['Solid'],width=0.0007, 
        label='Solid', color='light blue', alpha=0.6)
ax2.bar(df_phase_mix_osug2101.index, df_phase_mix_osug2101['Liquid'],width=0.0007, bottom=df_phase_mix_osug2101['Solid'],
       label='Liquid', color='light red', alpha=0.6)
ax2.bar(df_phase_mix_osug2101.index, df_phase_mix_osug2101['Mixture'],width=0.0007, bottom=df_phase_mix_osug2101['Solid']+df_phase_mix_osug2101['Liquid'],
       label='Mixture', color='light green', alpha=0.6)
#ax2.legend(loc=2)
ax2.set(xlabel=None, xticklabels=[])
ax2.set_ylabel('Particles/min')
ax2.yaxis.set_label_position("right")
ax2.yaxis.tick_right()
# move ticks
#ax2.tick_params(axis='y', which='both', labelleft=False, labelright=True)

#ax2.set_title('Inset of Elastic Region')
#ax2.set_xlim([0,0.008])
#ax2.set_ylim([0,100])

ax3 = ax1.inset_axes([ax1.get_xlim()[0], Stations.loc['SMU']['Elev'], ax1.get_xlim()[1]-ax1.get_xlim()[0], 400],
                     transform=ax1.transData)
ax3.set_facecolor('grey')
ax3.patch.set_alpha(0.2)
ax3.set_xlim(xlim[0],xlim[1])
ax3.set_ylim(0,1000)
#ax2.plot(pd.to_datetime(time[:].data, unit='s'),h_smooth)
ax3.bar(df_phase_mix_smu.index, df_phase_mix['Solid'],width=0.0007, 
        label='Solid', color='light blue', alpha=0.6)
ax3.bar(df_phase_mix_smu.index, df_phase_mix['Liquid'],width=0.0007, bottom=df_phase_mix_smu['Solid'],
       label='Liquid', color='light red', alpha=0.6)
ax3.bar(df_phase_mix_smu.index, df_phase_mix_smu['Mixture'],width=0.0007, bottom=df_phase_mix_smu['Solid']+df_phase_mix_smu['Liquid'],
       label='Mixture', color='light green', alpha=0.6)
#ax3.legend(loc=2)
ax3.set(xlabel=None, xticklabels=[])
#ax3.set_ylabel('Particles number')
ax3.yaxis.set_label_position("right")
ax3.yaxis.tick_right()

ax4 = ax1.inset_axes([ax1.get_xlim()[0], Stations.loc['Chamrousse']['Elev'], ax1.get_xlim()[1]-ax1.get_xlim()[0], 400],
                     transform=ax1.transData)
ax4.set_facecolor('grey')
ax4.patch.set_alpha(0.2)
ax4.set_xlim(xlim[0],xlim[1])
ax4.set_ylim(0,3500)
#ax2.plot(pd.to_datetime(time[:].data, unit='s'),h_smooth)
ax4.bar(df_phase_mix_chamrousse.index, df_phase_mix_chamrousse['Solid'],width=0.0007, 
        label='Solid', color='light blue', alpha=0.6)
ax4.bar(df_phase_mix_chamrousse.index, df_phase_mix_chamrousse['Liquid'],width=0.0007, bottom=df_phase_mix_chamrousse['Solid'],
       label='Liquid', color='light red', alpha=0.6)
ax4.bar(df_phase_mix_chamrousse.index, df_phase_mix_chamrousse['Mixture'],width=0.0007, bottom=df_phase_mix_chamrousse['Solid']+df_phase_mix_chamrousse['Liquid'],
       label='Mixture', color='light green', alpha=0.6)
#ax4.legend(loc=0)
ax4.legend(bbox_to_anchor=(0.95, 1.1), borderaxespad=0, 
           facecolor='grey', framealpha=0.2, edgecolor='k')
ax4.set(xlabel=None, xticklabels=[])
#ax4.set_ylabel('Particles number')
ax4.yaxis.set_label_position("right")
ax4.yaxis.tick_right()


fig.autofmt_xdate()
date_format = mdate.DateFormatter('%d%b \n %H:%M')
ax1.xaxis.set_major_formatter(date_format)
fig.tight_layout()
plt.rc('xtick', labelsize=10) 

#plt.show()
plt.savefig(path_save+'20210115_MRRvsParsivel_600dpi.png', format='png', dpi=600)