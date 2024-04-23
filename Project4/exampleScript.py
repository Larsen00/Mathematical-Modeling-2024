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
import os
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split

folder_path = 'Project4/processedfull'
files_in_directory = os.listdir(folder_path)
dates = [file.replace('.xlsx','') for file in files_in_directory if file.endswith('.xlsx')]

# Find all image files
file_name = []
for day in dates:
    file_name += glob.glob(f'{folder_path}/{day[6:8]}/*natural_color.npy')

# Point to production data
excel_str = [f'{folder_path}/{day}.xlsx' for day in dates]

# Load binary mask outlining Denmark
mask = np.load( folder_path + '/mask.npy')

# Allocate memory and load image data
Xdata = np.zeros((mask.sum(), len(file_name))) # X-variable: The values from the pixels in the images
groundIntensity = np.zeros(mask.sum())
times = []
timesDay = []
i = 0
for entry in file_name:
    img = np.load(entry)
    
    dummy = img[:,:,0]+img[:,:,1]-2*img[:,:,2]
    #dummy = dummy*mask
    Xdata[:,i] = dummy[mask].flatten()
    
    # It is assumed that the maximum value found of a pixel is the gound without a cloud
    groundIntensity = np.maximum(groundIntensity, Xdata[:,i])
    
    # Find time information in filename
    ind = entry.find('202403')
    
    
    times.append(entry[ind+8:ind+14])   # gices time of day hhmmss
    timesDay.append(entry[ind+6:ind+8]) # gives the date
    
    i +=1
    
timesDay = np.array(timesDay)
times = np.array(times)

def scale255(x):
    return (x + 255*2)/4

# groundIntensity = scale255(groundIntensity)
print(np.mean(groundIntensity))
print(np.median(groundIntensity))

plt.hist(groundIntensity, bins=20, color='blue', alpha=0.7)  # You can adjust the number of bins and color as needed
plt.title('Histogram of groundIntensity')
plt.xlabel('Intensity')
plt.ylabel('Frequency')
plt.show()


x = np.full(mask.shape, np.nan)
x[mask] = groundIntensity

np.savetxt("reshapedIntensity.csv", x, delimiter=",")


for i in range(len(file_name)):
    # Xdata[:, i] = scale255(Xdata[:, i]) / groundIntensity  # find en måde at shift det på (hvis nødvendit) + lav et filter
    Xdata[:, i] /= groundIntensity
    

# get target/production values
Y = []
for excel_file in excel_str:
    target = pd.read_excel(excel_file, usecols="A,F") # Minutes1DK, SolorPower
    target_times = target['Minutes1UTC']
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

X = Xdata.T
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.1, random_state=42)



model = Ridge()
model.fit(X_train,Y_train)
Y_test_hat = model.predict(X_test)
print(sum((Y_test_hat-Y_test)**2)/len(Y_test))
print('DONE')
