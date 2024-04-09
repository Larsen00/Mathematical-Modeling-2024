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
from sklearn.linear_model import Ridge

# Find all image files
file_name = glob.glob('Project4/processed/17/*natural_color.npy')
file_name = file_name + glob.glob('Project4/processed/18/*natural_color.npy')
file_name = file_name + glob.glob('Project4/processed/19/*natural_color.npy')
file_name = file_name + glob.glob('Project4/processed/26/*natural_color.npy')
file_name = file_name + glob.glob('Project4/processed/29/*natural_color.npy')
file_name = file_name + glob.glob('Project4/processed/31/*natural_color.npy')

# Point to production data
excel_str = ['Project4/processed/20240317.xlsx',
             'Project4/processed/20240318.xlsx',
             'Project4/processed/20240319.xlsx',
             'Project4/processed/20240326.xlsx',
             'Project4/processed/20240329.xlsx',
             'Project4/processed/20240331.xlsx']

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
    
    
    times.append(entry[ind+8:ind+14])
    timesDay.append(entry[ind+6:ind+8])
    
    i +=1
    
timesDay = np.array(timesDay)
times = np.array(times)
    
# get target/production values
Y = []
for excel_files in excel_str:
    target = pd.read_excel(excel_files, usecols="B,F")
    target_times = target['Minutes1DK']
    # Ensure the column is in datetime format
    target_times = pd.to_datetime(target_times)
    # Format the time to HHMMSS and remove colons
    target_times = target_times.dt.strftime('%H%M%S')    
    
    for entry in times[timesDay==excel_files[-7:-5]]:
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
X = Xdata.T
model = Ridge()
model.fit(X,Y)
model.predict(X)
print('DONE')