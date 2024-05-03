import numpy as np
import pandas as pd
import os
from sklearn.linear_model import RidgeCV, Ridge
from flow_ver3_1 import interpolate_flow, Lucas_Kanade_method, extrapolate_flow
from load_images import load_images
import matplotlib.pyplot as plt
from datetime import datetime
import matplotlib.dates as mdates
from sklearn.preprocessing import StandardScaler
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

        X_T = np.zeros((mask.sum(), n_images)) # X_T is the transpose of X, X_T has (pixels, time) dimensions
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

def ridge_model_on_extrapolated_data(dates, test_date, alphas, target_time:str, interpolation=False):
    """
    Do ridge model on extrapolated data
    ---
    Args:
        Given the dates, test_date, alphas, and target_time
    Return: 
        errors and average distances.
    """
    date_index = dates.index(test_date)

    # load data and labels from all dates other than the test date
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
    # load data and labels before the target_time at the test date
    X_temp, Y_temp = get_training_data_and_labels(dates[date_index], path, interpolation=False, time_until=target_time)


    X = np.vstack([X,X_temp])
    Y = np.hstack([Y,Y_temp])
    if interpolation == True:
        X_interpolated, Y_interpolated = get_interpolated_data_and_labels(X, Y)
        X = np.vstack([X, X_interpolated])
        Y_original = Y.copy()
        Y = np.hstack([Y, Y_interpolated])
        
    
    scaler = StandardScaler()
    X_standardized = scaler.fit_transform(X)

    # Train the ridge and select the model with the best regularization parameter alpha_
    model = RidgeCV(alphas=alphas, store_cv_values=True)
    model.fit(X_standardized, Y)
    print("optimal alpha is ", model.alpha_)
    # find the optimal alpha and train again
    model = RidgeCV(model.alpha_)
    model.fit(X_standardized,Y)
    
    # Do extrapolation from the target_time of the test date, and predict the power
    V, timesDay, times, mask = load_images(test_date, path)
    print(timesDay, times)
    
    # Define how many objects subject to calculating optical flow(from the)
    time_index = len(times[times <= target_time]) - 1     # last known time index
    objects = range(time_index, len(times))
    dps_images = Lucas_Kanade_method(V, objects=objects)
    minutes_after = 15
    V_extrapolation, timestamps = extrapolate_flow(dps_images, V, timesDay, times, mask, minutes_after=minutes_after, objects=objects, show=False)

    # For every extrapolated image, predict the power and register the error
    errors = []
    average_distances = []
    y_prediction_list = []
    y_true_list = []
    true_time_list = []
    for i in range(V_extrapolation.shape[2]):
        X_test_standardized = scaler.transform(np.expand_dims(V_extrapolation[:,:,i][mask],0)) # special purpose, the input should be a one row long 2D array
        y_prediction = model.predict(X_test_standardized)
        y_prediction_list.append(y_prediction)
        # get the true power from the excel file
        excel_file = f'{path}/{test_date}.xlsx'
        excel_data = pd.read_excel(excel_file, usecols="A,F").values    # this is a numpy array first col with timestamp format, second col with numbers
        excel_times, excel_power = return_time_and_power_from_excel_data(excel_data)
        true_time = round_seconds([timestamps[i]])[0]   # strange but acceptable
        indices = np.where(excel_times == true_time)[0] # this is the index of the time in the excel file in a array, if there are mutiple values,then there are time duplicates
        y_true = excel_power[indices]
        y_true_list.append(y_true)

        error = MSE(y_prediction, y_true)
        errors.append(error)

        average_distance = average_distance(y_prediction, y_true)
        average_distances.append(average_distance)

        print(f"Error at time {true_time} is {error}")
        print(f"Average distance at time {true_time} is {average_distance}")
        true_time_list.append(true_time)
        
        # append new image to the training data
        if i != V_extrapolation.shape[2] - 1:    # prevent overflow
            X, Y = np.vstack([X, np.expand_dims(V[:,:,time_index + i + 1][mask], axis=0)]), np.hstack([Y, y_true])
            if interpolation == True:
                tmpX, tmpY = np.vstack([V[:,:,time_index + i][mask], V[:,:,time_index + i + 1][mask]]), np.hstack([Y_original[time_index + i], y_true])
                X_interpolated, Y_interpolated = get_interpolated_data_and_labels(tmpX, tmpY)
                X = np.vstack([X, X_interpolated])
                Y = np.hstack([Y, Y_interpolated])
            X_standardized = scaler.fit_transform(X)
            model.fit(X_standardized,Y)
    
    return errors, average_distances, y_prediction_list, y_true_list, true_time_list

def MSE(Y_pred, Y_true):
    return np.mean((Y_pred-Y_true)**2)

def average_distance(Y_pred, Y_true):
    return np.mean((Y_pred-Y_true))

def weighted_average_interpolation(X1, X2, weight):
    return X1*weight + X2*(1-weight)

def simple_mean_interpolation(X1, X2):
    return (X1 + X2) / 2

def get_interpolated_data_and_labels(X, Y):
    n = X.shape[0] - 1
    X_interpolated = np.zeros((n, X.shape[1]))
    Y_interpolated = np.zeros(n)
    for i in range(n):
        X_interpolated[i, :] = simple_mean_interpolation(X[i, :], X[i + 1, :])
        Y_interpolated[i] = simple_mean_interpolation(Y[i], Y[i + 1])
    return X_interpolated, Y_interpolated

def model_performance_test(dates):
    print("model performance test")
    alphas = np.logspace(-6, 6, 200)
    #################################
    # same result as get_training_data_and_labels
    X = np.load("X.npy")
    Y = np.load("Y.npy")

    scaler = StandardScaler()

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=2)

    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)


    # Create the RidgeCV model
    m = RidgeCV(alphas=alphas)

    # Fit the model to the training data
    m.fit(X_train, Y_train)

    # The optimal alpha value
    optimal_alpha = m.alpha_
    print(f'Optimal alpha: {optimal_alpha}')

    # Now create and fit the Ridge regression model with the optimal alpha
    model = Ridge(alpha=optimal_alpha)
    model.fit(X_train, Y_train)
    Y_prediction = model.predict(X_test)
    residuals = Y_prediction - Y_test
    print(np.mean(residuals**2))
    plt.scatter(Y_prediction, residuals)
    plt.xlabel('Predicted values')
    plt.ylabel('Residuals')
    plt.axhline(y=0, color='r', linestyle='--')
    plt.title("Residuals for model without interpolation")
    plt.savefig("Residuals_model_without_interpolation.svg")
    plt.clf()
    #############################
    X = np.load("X.npy")
    Y = np.load("Y.npy")

    scaler = StandardScaler()

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=2)

    X_interpolated, Y_interpolated = get_interpolated_data_and_labels(X, Y)
    X_train = scaler.fit_transform(np.vstack((X_train, X_interpolated)))
    Y_train = np.hstack((Y_train, Y_interpolated))

    X_test = scaler.transform(X_test)

    # Create the RidgeCV model
    m = RidgeCV(alphas=alphas)

    # Fit the model to the training data
    m.fit(X_train, Y_train)

    # The optimal alpha value
    optimal_alpha = m.alpha_
    print(f'Optimal alpha: {optimal_alpha}')

    # Now create and fit the Ridge regression model with the optimal alpha
    model = Ridge(alpha=optimal_alpha)
    model.fit(X_train, Y_train)
    Y_prediction = model.predict(X_test)
    residuals = Y_prediction - Y_test
    print(np.mean(residuals**2))
    plt.scatter(Y_prediction, residuals)
    plt.xlabel('Predicted values')
    plt.ylabel('Residuals')
    plt.axhline(y=0, color='r', linestyle='--')
    plt.title("Residuals for model with simple weighted interpolation")
    plt.savefig("Residuals_model_with_simple_weighted_interpolation.svg")
    plt.clf()
    ########################################################################
    
    # it takes too long to run the following code
    print("Warning: the following code takes too long to run")
    scaler = StandardScaler()
    for i, date in enumerate(dates):
        X_temp, Y_temp = get_training_data_and_labels(date, path, interpolation=True)
        print(f"Date: {date}, data shape: {X_temp.shape}, label shape: {Y_temp.shape}")
        if i == 0:
            X = X_temp
            Y = Y_temp
        else:
            X = np.vstack([X,X_temp])
            Y = np.hstack([Y,Y_temp])
    
    
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=2)
    
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Create the RidgeCV model
    m = RidgeCV(alphas=alphas)

    # Fit the model to the training data
    m.fit(X_train, Y_train)

    # The optimal alpha value
    optimal_alpha = m.alpha_
    print(f'Optimal alpha: {optimal_alpha}')

    # Now create and fit the Ridge regression model with the optimal alpha
    model = Ridge(alpha=optimal_alpha)
    model.fit(X_train, Y_train)
    Y_prediction = model.predict(X_test)
    residuals = Y_prediction - Y_test
    print(np.mean(residuals**2))
    plt.scatter(Y_prediction, residuals)
    plt.xlabel('Predicted values')
    plt.ylabel('Residuals')
    plt.axhline(y=0, color='r', linestyle='--')
    plt.title("Residuals model with optical flow interpolation")
    plt.savefig("Residuals_model_with_optical_flow_interpolation.svg")
    return

if __name__ == '__main__':
    path = 'Project4/Processedfull'
    files_in_directory = os.listdir(path)
    dates = [file.replace('.xlsx','') for file in files_in_directory if file.endswith('.xlsx')]
    sort_dates(dates)

    # actually it just plot all dates
    model_performance_test(dates)
        