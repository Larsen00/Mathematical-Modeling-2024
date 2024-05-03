import numpy as np
import pandas as pd
import os
from sklearn.linear_model import RidgeCV
from flow_ver3_1 import interpolate_flow, Lucas_Kanade_method, extrapolate_flow
from load_images import load_images
from sklearn.model_selection import train_test_split

def remove_dates(dates, string_list):
   return [d for d in dates if d  not in string_list] ####change back return [d for d in dates if d not in string_list]

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

def time_after_some_minutes(time:str, minutes:int):
    """
    Given a time in string format 'hhmmss', return the time after some minutes in string format 'hhmmss'.
    """
    hour = int(time[:2])
    minute = int(time[2:4])
    second = int(time[4:6])
    minute += minutes
    if minute >= 60:
        hour += 1
        minute -= 60
    return f"{str(hour).zfill(2)}{str(minute).zfill(2)}{str(second).zfill(2)}"

def get_training_data_and_labels(date, path, interpolation=False, time_until=None):
    """
    given one date, return the training data and labels.
    """
    
    # Find all image files(V has (y,x,t) dimensions)
    V, timesDay, times, mask = load_images(date, path)
    if time_until != None and type(time_until) == str and len(time_until) == 6:
        times = times[times <= time_until]
        timesDay = timesDay[timesDay <= time_until]
        V = V[:, :, :len(times)]

    times = round_seconds(times)
    if interpolation == False:
        X_T = np.zeros((mask.sum(), V.shape[2])) # X_T is the transpose of X, X_T has (pixels, time) dimensions
        for i in range(V.shape[2]):
            X_T[:, i] = V[:, :, i][mask]
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
            X_T[:, i] = V_interpolation[:, :, i][mask]
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

def ridge_model_on_extrapolated_data(dates, test_date, alphas, target_time:str):
    
     
    date_index = dates.index(test_date)
    for i in range(date_index):
        date = dates[i]
        X_temp, Y_temp = get_training_data_and_labels(date, path, interpolation=False)
        print(f"Date: {date}, data shape: {X_temp.shape}, label shape: {Y_temp.shape}")
        if i == 0:
            X = X_temp
            Y = Y_temp
        else:
            X = np.vstack([X,X_temp])
            Y = np.hstack([Y,Y_temp])
    X_temp, Y_temp = get_training_data_and_labels(dates[date_index], path, interpolation=True, time_until=target_time)
    X = np.vstack([X,X_temp])
    Y = np.hstack([Y,Y_temp])
    model = RidgeCV(alphas=alphas, store_cv_values=True)
    
    model.fit(X, Y)
    print("optimal alpha is ", model.alpha_)
    # find the optimal alpha and train again
    model = RidgeCV(model.alpha_)
    model.fit(X,Y)
    
    V, timesDay, times, mask = load_images(test_date, path)
    print(timesDay, times)
    # Define how many objects subject to calculating optical flow
    time_index = times.index[target_time]
    objects=range(time_index, len(times))
    dps_images = Lucas_Kanade_method(V, objects=objects)
    minutes_after = 15
    V_extrapolation, timestamps = extrapolate_flow(dps_images, V, timesDay, times, mask, minutes_after=minutes_after, objects=objects, show=False)
    errors = []
    average_distances = []
    for i in range(V_extrapolation.shape[2]):
        y_prediction = model.predict(V_extrapolation[:,:,i][mask])
        excel_file = f'{path}/{test_date}.xlsx'
        excel_data = pd.read_excel(excel_file, usecols="A,F").values    # this is a numpy array first col with timestamp format, second col with numbers
        excel_times, excel_power = return_time_and_power_from_excel_data(excel_data)
        true_time = timestamps[i]
        indices = np.where(excel_times == true_time)[0] # this is the index of the time in the excel file in a array, if there are mutiple values,then there are time duplicates
        y_true = excel_power[indices]
        

        error = MSE(y_prediction, y_true)
        errors.append(error)

        average_distance = averge_distance(y_prediction, y_true)
        average_distances.append(average_distance)

        print(f"Error at time {true_time} is {error}")
        
        # append new image to the training data
        if i != V_extrapolation.shape[2] -1:    # prevent overflow
            X, Y = np.vstack([X, V[:,:,time_index + i + 1]]), np.hstack([Y, y_true])
            model.fit(X,Y)
    return errors, average_distances

def MSE(Y_pred, Y_true):
    return np.mean((Y_pred-Y_true)**2)

def averge_distance(Y_pred, Y_true):
    return np.mean((Y_pred-Y_true))

if __name__ == '__main__':
    path = 'Project4/Processedfull'
    files_in_directory = os.listdir(path)
    dates = [file.replace('.xlsx','') for file in files_in_directory if file.endswith('.xlsx')]
    dates_to_remove =['20240306','20240307','20240308','20240309','20240310','20240311','20240312','20240313','20240314','20240317','20240318','20240319','20240326','20240329','20240331']
    dates_remain = remove_dates(dates, dates_to_remove)
    sort_dates(dates_remain)
    sort_dates(dates)
    print(dates_remain)

    # for i, date in enumerate(dates_remain):
    #     X_temp, Y_temp = get_training_data_and_labels(date, path, interpolation=False)
    #     print(f"Date: {date}, data shape: {X_temp.shape}, label shape: {Y_temp.shape}")
    #     if i == 0:
    #         X = X_temp
    #         Y = Y_temp
    #     else:
    #         X = np.vstack([X,X_temp])
    #         Y = np.hstack([Y,Y_temp])
    # 
    # number_of_alphas = 1000
    # alpha_vals = np.logspace(-4, 4, number_of_alphas)
    # cv_vals = np.zeros((0, number_of_alphas))
    # print("Creating model")
    # model = RidgeCV(alpha_vals, store_cv_values=True)
    # model.fit(X, Y)
    # # cv_vals = np.vstack((cv_vals, model.cv_values_))    # ver.1 model.cv_values_ is a 2D array with shape (n_samples, n_alphas)
    # cv_vals = model.cv_values_.mean(axis=0)    # ver.2 efficiency consideration: model.cv_values_ is a 2D array with shape (n_samples, n_alphas)
    # print("optimal alpha is ", model.alpha_, " and tis cv value is ", np.min(cv_vals))
# 
    # X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.1, random_state=2)
    # model = RidgeCV(model.alpha_)
    # model.fit(X_train,Y_train)
    # Y_pred = model.predict(X_test)
    # 
    # error_pred = (np.mean((Y_pred-Y_test)**2))
    # 
    # avg_diff = np.mean((Y_pred-Y_test))
# 
    # print(f'MSE of prediction: {error_pred}')
    # print(f'avg_diff not absed {avg_diff}')
    # Y_baseline = np.mean(Y_train)
    # error_baseline = (np.mean((np.ones_like(Y_test)*Y_baseline*-Y_test)**2))
    # print(f'MSE of baseline: {error_baseline}')

    time_until = "120000"
    number_of_alphas = 1000
    alpha_vals = np.logspace(-4, 4, number_of_alphas)
    test_date= dates_to_remove[0]
    ridge_model_on_extrapolated_data(dates, test_date, alpha_vals, time_until)

    pass
        