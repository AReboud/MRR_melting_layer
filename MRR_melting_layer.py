# -*- coding: utf-8 -*-
"""
Created on Sun Jun  5 23:40:18 2022

@author: Arnaud Reboud (IGE)
"""

#Commons
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
import func

MRR_file = './Data/MRR/OSUG-B_20201212.nc'
dir_plot = './Plots/MRR/'

#Parameters and constants
alt_station = 230 #m.a.s.l : elevation of OSUG-B MRR station
chi = 0.7
r = 1
sigma = 20
time_interp = 90 #min

event_start = '2020-12-12 04:00:00'
event_stop = '2020-12-12 22:00:00'

#compute and get the melting layer elevations of the event
ML_Ze_height, ML_Ze_top, ML_Ze_bot, ML_W_height, ML_W_top, ML_W_bot, time = func.get_MRR_melting_layer(MRR_file,event_start,event_stop,
                       path_dir_out=dir_plot, alt_station=alt_station)

#plot the melting layer elevations
func.plot_MRR_melting_layer(ML_Ze_height, ML_Ze_top, ML_Ze_bot, 
                            ML_W_height, ML_W_top, ML_W_bot, time,
                            event_start, event_stop, alt_station, path_dir_out=dir_plot)

#Specific time inside the event time range
sat_time ='2020-12-12 10:00:00'

#plot the verticale profiles of reflectivity and Doppler velocity at the specified time
func.plot_spec_time(spec_time=sat_time,MRRncfile=MRR_file,event_start=event_start,
                    event_stop=event_stop, path_dir_out=dir_plot)

#get the melting layer elevations at the specified time
ML_elevation_list = func.get_spec_melting_layer(spec_time=sat_time,ML_Ze_height=ML_Ze_height, ML_Ze_top=ML_Ze_top, ML_Ze_bot=ML_Ze_bot, 
                                                ML_W_height=ML_W_height, ML_W_top=ML_W_top, ML_W_bot=ML_W_bot, time=time, alt_station= alt_station)

print(ML_elevation_list)