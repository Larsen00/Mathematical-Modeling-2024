import numpy as np
import pandas as pd
import os
from sklearn.linear_model import RidgeCV
from Project4.flow_ver3_1 import interpolate_flow, Lucas_Kanade_method
from Project4.load_images import load_images


def sort_dates(dates):
    """
    Sorts the dates in ascending order.
    """
    # Sort the file names in time
    ind = dates[0].find('202403')
    dates.sort(key=lambda x: int(x[ind+6:ind+8]))
    return

def round_seconds(times):
    """
    Rounds the time format 'hhmmss' to the nearest minute 'hhmm00'.
    """
    result = []
    for time in times:
        if int(time[4:6]) > 30:
            time = time[:2] + str(int(time[2:4]) + 1).zfill(2) + '00'
        else:
            time = time[:4] + '00'
        # round hours if minutes was 59 like 015959 -> 020000
        if time[2:4] == '60':
            time = str(int(time[:2]) + 1).zfill(2) + '00'
        result.append(time)
    return result

def convert_pddatetime_to_strtime(datetime):
    """
    Converts the datetime format 'YYYY-MM-DDThh:mm:ss' to the format 'hh:mm:ss'.
    """
    tmp = np.datetime_as_string(datetime.to_numpy())
    ind = tmp.find('T')
    return tmp[ind+1:ind+3] + tmp[ind+4:ind+6] + tmp[ind+7:ind+9]

def return_time_and_power_from_excel_data(excel_data):
    """
    Reads the excel data and returns the time and power values.
    """
    times = []
    for i in range(excel_data.shape[0]):
        times.append(convert_pddatetime_to_strtime(excel_data[i,0]))
    times = np.array(times)
    power = excel_data[:,1]
    return times, power

def warning_message(s):
    """
    Prints a warning message.
    """
    print(f"Warning: {s}")
    return

def get_training_data_and_labels(date, path, interpolation=False):
    """
    given one date, return the training data and labels.
    """
    
    # Find all image files(V has (y,x,t) dimensions)
    V, timesDay, times, mask = load_images(date, path)
    times = round_seconds(times)
    if interpolation == False:
        X_T = np.zeros((V.shape[0]*V.shape[1], V.shape[2])) # X_T is the transpose of X, X_T has (pixels, time) dimensions
        for i in range(V.shape[2]):
            X_T[:, i] = V[:, :, i].flatten()
        X = X_T.T

        Y = np.zeros(V.shape[2])
        excel_file = f'{path}/{date}.xlsx'
        excel_data = pd.read_excel(excel_file, usecols="A,F").values    # this is a numpy array first col with timestamp format, second col with numbers
        excel_times, excel_power = return_time_and_power_from_excel_data(excel_data)
        for i, time in enumerate(times):  # times are from images
            indices = np.where(excel_times == time)[0] # this is the index of the time in the excel file in a array, if there are mutiple values,then there are time duplicates
            if indices.size > 1:
                warning_message(f"Multiple values of time {time} in the excel file.")
            Y[i] = excel_power[indices[0]]
    else:
        n_interpolation_points_between_images = 3
        n_images = (V.shape[2]-1)*n_interpolation_points_between_images + 1    # interpolate every minute, thus if there is 2 images, then there are (2-1)15+1=16 images
        objects = range(V.shape[2]-1) # quite special here
        dps_images = Lucas_Kanade_method(V, objects=objects)
        V_interpolation, timestamps = interpolate_flow(dps_images, V, timesDay, times, mask, objects=objects, show=False, n=n_interpolation_points_between_images)

        # add the last image
        V_interpolation = np.concatenate((V_interpolation, np.expand_dims(V[:,:,-1], 2)), axis=2)
        timestamps.append(times[-1])

        X_T = np.zeros((V.shape[0]*V.shape[1], n_images)) # X_T is the transpose of X, X_T has (pixels, time) dimensions
        for i in range(n_images):
            X_T[:, i] = V_interpolation[:, :, i].flatten()
        X = X_T.T

        Y = np.zeros(n_images)
        excel_file = f'{path}/{date}.xlsx'
        excel_data = pd.read_excel(excel_file, usecols="A,F").values    # this is a numpy array first col with timestamp format, second col with numbers
        excel_times, excel_power = return_time_and_power_from_excel_data(excel_data)
        for i, time in enumerate(times):  # times are from images
            indices = np.where(excel_times == time)[0] # this is the index of the time in the excel file in a array, if there are mutiple values,then there are time duplicates
            if indices.size > 1:
                warning_message(f"Multiple values of time {time} in the excel file.")
            Y[i] = excel_power[indices[0]]
        pass
        
    return X, Y
    

def print_lowest_cv_result(alphas, cv_vals, weights):
    """
    Prints the cross-validation values.
    """
    cv_per_alpha = np.average(cv_vals, axis=0, weights=weights)
    index = np.argmin(cv_per_alpha)
    print(f"alpha which gives lowest cv is {alphas[index]}, cv value is {cv_per_alpha[index]}")
    
    return

if __name__ == '__main__':
    number_of_alphas = 10000
    alpha_vals = np.logspace(-4, 4, number_of_alphas)

    path = 'Project4/Processedfull'
    files_in_directory = os.listdir(path)
    dates = [file.replace('.xlsx','') for file in files_in_directory if file.endswith('.xlsx')]
    sort_dates(dates)
    print(dates)
    cv_vals = np.zeros((0, number_of_alphas))
    weights = []
    for i, date in enumerate(dates):
        X, Y = get_training_data_and_labels(date, path, interpolation=True)
        print(f"Date: {date}, data shape: {X.shape}, label shape: {Y.shape}")
        model = RidgeCV(alpha_vals, store_cv_values=True)
        model.fit(X, Y)
        # cv_vals = np.vstack((cv_vals, model.cv_values_))    # ver.1 model.cv_values_ is a 2D array with shape (n_samples, n_alphas)
        cv_vals = np.vstack((cv_vals, model.cv_values_.mean(axis=0)))    # ver.2 efficiency consideration: model.cv_values_ is a 2D array with shape (n_samples, n_alphas)
        weights.append(Y.size)

    print_lowest_cv_result(alpha_vals, cv_vals, weights)
    pass
        

    pass