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
from sklearn.model_selection import train_test_split


# dates of images
dates = ['0317', '0318', '0319', '0326', '0329', '0331']
# Function to calculate "closeness" to midday
def calculate_closeness_month(month):
    # Assuming midday is the peak and values decrease towards 6 AM and 6 PM
    # Using a simple model where the maximum closeness is at 12 PM
    # and minimum at 6 AM and 6 PM. This uses a Gaussian distribution concept.
    return np.exp(-((month - 12)**2) / (2 * (3**2)))  # Standard deviation is 3 hours for a sharper drop
def calculate_closeness_hour(hour,month): # month used to determine standard deviation - winter = shorter days
    # Assuming midday is the peak and values decrease towards 6 AM and 6 PM
    # Using a simple model where the maximum closeness is at 12 PM
    # and minimum at 6 AM and 6 PM. This uses a Gaussian distribution concept.
    return np.exp(-((hour - 12)**2) / (2 * ((0.38/calculate_closeness_month(month))**2)))  # Standard deviation is 3 hours for a sharper drop

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

#Just used for testing
closeness_values_hour =1
closeness_values_month =1

for entry in file_name:
    img = np.load(entry)
    # Find time information in filename
    ind = entry.find('202403')
    times_new = entry[ind+8:ind+14]
    timesDay_new = entry[ind+6:ind+8]

    dummy = (img[:,:,0]+img[:,:,1]-2*img[:,:,2])[mask].flatten()

    #standardize data
    if np.std(dummy) != 0:
        dummy = (dummy-np.mean(dummy))/np.std(dummy)

    hour = (int(times_new[:2]))
    month = int(entry[ind+4:ind+6])*2 ## because then months have same role as time of day, a bit rough estimate
    closeness_values_hour = calculate_closeness_hour(hour,month)
    closeness_values_month = calculate_closeness_month(month)
    #dummy = (img[:,:,0]+img[:,:,1]-2*img[:,:,2])
    dummy = dummy*closeness_values_hour*closeness_values_month
    print(times_new,timesDay_new,closeness_values_hour,closeness_values_month)
    #MSE results on test data of normalizing with hour of day and month of year on one data split:
    #BTW gausian distribution is with std = 3
    #No normalization: 95784
    #Hour normalization: 34791
    #Month normalization:88340
    #Month and hour normalization: 22996
    #Month and hour normalization + ground intensity: 33978
    Xdata[:,i] = dummy
    


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
X = Xdata.T
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.1, random_state=1)



model = Ridge()
model.fit(X_train,Y_train)
Y_test_hat = model.predict(X_test)
print((np.round(np.mean((Y_test_hat-Y_test)**2))))
print('DONE')
