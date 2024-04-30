
import openpyxl
#from flow_ver3 import *
import glob
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge, RidgeCV
from sklearn.model_selection import train_test_split

print("Loading data")
path = 'Project4/Processedfull'
files_in_directory = os.listdir(path)
dates = [file.replace('.xlsx','') for file in files_in_directory if file.endswith('.xlsx')]

def remove_dates(dates, string_list):
   return [d for d in dates if d  not in string_list] ####change back return [d for d in dates if d not in string_list]

# remove some dates
dates_to_remove =['20240317']
dates = remove_dates(dates, dates_to_remove)

# Function to calculate "closeness" to midday
def hour_func(hour): # month used to determine standard deviation - winter = shorter days
    #This uses a Gaussian distribution concept.
    return np.exp(-((hour - 12)**2) / (((4)**2)))  # Standard deviation is 4.5 hours for a sharper drop

# Find all image files
file_name = []
file_name_removed = []
for day in dates:
    file_name += glob.glob(f'{path}/{day[6:8]}/*natural_color.npy')
for day in dates_to_remove:
    file_name_removed += glob.glob(f'{path}/{day[6:8]}/*natural_color.npy')

# Point to production data
excel_str = [f'{path}/{day}.xlsx' for day in dates]

# Load binary mask outlining Denmark
mask = np.load( path + '/mask.npy')

# Allocate memory and load image data
Xdata = np.zeros((mask.sum(), len(file_name))) # X-variable: The values from the pixels in the images
times = []
timesDay = []
i = 0


for entry in file_name:
    img = np.load(entry)
    # Find time information in filename
    ind = entry.find('202403')
    times_new = entry[ind+8:ind+14]
    timesDay_new = entry[ind+6:ind+8]

    dummy = (img[:,:,0]+img[:,:,1]-2*img[:,:,2])[mask].flatten()
    minutes = float(times_new[2:4])/60
    hour = float(times_new[:2]) + minutes
    hour_factor= hour_func(hour)
    dummy = dummy*hour_factor
    Xdata[:,i] = dummy
    
    times.append(entry[ind+8:ind+14])   # gices time of day hhmmss
    timesDay.append(entry[ind+6:ind+8]) # gives the date
    i +=1
    
timesDay = np.array(timesDay)
times = np.array(times)


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
        Y.append(dummy.values[0])
print('DONE')
X = Xdata.T
alpha_values = np.linspace(1, 100000, 1000)  # Example range from very small to large alphas
model = RidgeCV(alphas=alpha_values, store_cv_values=True)  
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.1, random_state=2)
model.fit(X_train,Y_train)
model.predict(X_test)
print(model.alpha_)
model = Ridge(alpha=model.alpha_)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.1, random_state=4)
model.fit(X_train,Y_train)
Y_pred = model.predict(X_test)
error = ((np.mean((Y_pred-Y_test)**2)))
avg_diff = np.mean(np.abs(Y_pred-Y_test))

print(np.mean(error))
print(avg_diff)
model.fit(X,Y)