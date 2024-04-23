
import openpyxl
import glob
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split



folder_path = 'Project4/processedfull'
files_in_directory = os.listdir(folder_path)
dates = [file.replace('.xlsx','') for file in files_in_directory if file.endswith('.xlsx')]


# Function to calculate "closeness" to midday
def month_func(month):
    # Assuming june,july is the peak and values decrease. This uses a Gaussian distribution concept.
    return np.exp(-((month - 6.5)**2) / (2 * (6**2)))  # Standard deviation is 4 hours for a sharper drop
def hour_func(hour,month): # month used to determine standard deviation - winter = shorter days
    #This uses a Gaussian distribution concept.
    return np.exp(-((hour - 12)**2) / (((month_func(month)*6)**2)))  # Standard deviation is 5 hours for a sharper drop

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

    hour = (int(times_new[:2]))+1
    month = int(entry[ind+4:ind+6])
    hour_factor= hour_func(hour,month)
    month_factor = month_func(month)
    #dummy = (img[:,:,0]+img[:,:,1]-2*img[:,:,2])
    dummy = dummy*hour_factor*month_factor
    print(times_new,timesDay_new,hour_factor,month_factor,month)
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
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.1, random_state=1)



model = Ridge()
model.fit(X_train,Y_train)
Y_test_hat = model.predict(X_test)
print((np.round(np.mean((Y_test_hat-Y_test)**2))))
print(np.mean(abs(Y_test_hat-Y_test)))
