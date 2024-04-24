
import openpyxl
import glob
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge, RidgeCV
from sklearn.model_selection import train_test_split

print("Loading data")
folder_path = 'Project4/Processedfull'
files_in_directory = os.listdir(folder_path)
dates = [file.replace('.xlsx','') for file in files_in_directory if file.endswith('.xlsx')]

def remove_dates(dates, string_list):
   return [d for d in dates if d not in string_list]

# remove some dates
dates_to_remove =['20240317','20240318']
dates = remove_dates(dates, dates_to_remove)

# Function to calculate "closeness" to midday
def month_func(month):
    # Assuming june,july is the peak and values decrease. This uses a Gaussian distribution concept.
    return 1#np.exp(-((month - 5.5)**2) / (2 * (5.5**2)))  # Standard deviation is 5.5 hours for a sharper drop
def hour_func(hour): # month used to determine standard deviation - winter = shorter days
    #This uses a Gaussian distribution concept.
    return np.exp(-((hour - 12)**2) / (((5)**2)))  # Standard deviation is 4.5 hours for a sharper drop

# Find all image files
file_name = []
file_name_removed = []
for day in dates:
    file_name += glob.glob(f'{folder_path}/{day[6:8]}/*natural_color.npy')
for day in dates_to_remove:
    file_name_removed += glob.glob(f'{folder_path}/{day[6:8]}/*natural_color.npy')
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
    if int(times_new[2:4]) > 30:
        hour = (int(times_new[:2]))+1
    else: hour = (int(times_new[:2]))
    month = int(entry[ind+4:ind+6])
    hour_factor= hour_func(hour)
    month_factor = month_func(month)
    #dummy = (img[:,:,0]+img[:,:,1]-2*img[:,:,2])
    dummy = dummy*hour_factor*month_factor
    
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
errors=[]
avg_distances=[]
alpha_values = np.logspace(-6, 6, 13)  # Example range from very small to large alphas
# Initialize RidgeCV
opt_alphas = []
for i in range(0,5): 
    model = RidgeCV(alphas=alpha_values, store_cv_values=True)  
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.1, random_state=2*i)
    model.fit(X_train,Y_train)
    Y_test_hat = model.predict(X_test)
    error = ((np.mean((Y_test_hat-Y_test)**2)))
    avg_dist = np.mean(np.abs(Y_test_hat-Y_test))
    errors.append(error)
    avg_distances.append(avg_dist)
    opt_alphas.append(model.alpha_)
print(opt_alphas)
model = Ridge(alpha=2000)
print(np.mean(errors))
print(avg_dist)

model.fit(X,Y)
