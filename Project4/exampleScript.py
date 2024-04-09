# -*- coding: utf-8 -*-
"""
Created on Tue Apr  9 00:57:17 2024

@author: anym
"""
import openpyxl
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# dates of images
dates = ['0317', '0318', '0319', '0326', '0329', '0331']

# Find all image files
file_name = []
for day in dates:
    file_name += glob.glob(f'Project4/processed/{day[2:4]}/*natural_color.npy')

# Point to production data
excel_str = [f'Project4/processed/2024{day}.xlsx' for day in dates]

# Load binary mask outlining Denmark
mask = np.load('Project4/processed/mask.npy')

# Allocate memory and load image data
Xdata = np.zeros((mask.sum(), len(file_name))) # X-variable: The values from the pixels in the images
times = []
timesDay = []
i = 0
for entry in file_name:
    img = np.load(entry)
    
    dummy = img[:,:,0]+img[:,:,1]-2*img[:,:,2]
    #dummy = dummy*mask
    Xdata[:,i] = dummy[mask].flatten()
    
    # Find time information in filename
    ind = entry.find('202403')
    
    
    times.append(entry[ind+8:ind+14])   # gices time of day hhmmss
    timesDay.append(entry[ind+6:ind+8]) # gives the date
    
    i +=1
    
timesDay = np.array(timesDay)
times = np.array(times)
    
# get target/production values
Y = []
for excel_file in excel_str:
    target = pd.read_excel(excel_file, usecols="B,F") # Minutes1DK, SolorPower
    target_times = target['Minutes1DK']
    # Ensure the column is in datetime format
    target_times = pd.to_datetime(target_times)
    # Format the time to HHMMSS and remove colons
    target_times = target_times.dt.strftime('%H%M%S')    
    
    for entry in times[timesDay==excel_file[-7:-5]]:  # every times where on the same day as excelfile
        # Find rows where 'column_name' equals 'value_to_find'
        condition = target_times == entry[:-2] + '00'
        ind = target_times.index[condition].tolist()
        if len(ind)==0:
            # try minute before
            newTime = entry[:-2] + '00'
            dummy = int(newTime[-3]) -1
            newTime = newTime[:-3] + str(dummy) + newTime[-2:]
            condition = target_times == newTime 
            ind = target_times.index[condition].tolist()
            dummy = target['SolarPower'].iloc[ind]
        else:
            dummy = target['SolarPower'].iloc[ind]
        
        Y.append(dummy.values)
    
Y = np.array(Y)
print('DONE')