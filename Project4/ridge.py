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

def simple_mean_interpolation(X1, X2):
    """
    Performs simple mean calculation between two values.

    Parameters:
    X1 (float): The first value.
    X2 (float): The second value.

    Returns:
    float: The interpolated value.

    """
    # Calculate the mean of X1 and X2
    interpolated_value = (X1 + X2) / 2
    return interpolated_value


def get_interpolated_data_and_labels(X, Y):
    """
    Interpolates the data and labels using simple mean interpolation.

    Parameters:
    X (numpy.ndarray): The input data array of shape (n+1, m), where n is the number of data points and m is the number of features.
    Y (numpy.ndarray): The input labels array of shape (n+1,), where n is the number of data points.

    Returns:
    X_interpolated (numpy.ndarray): The interpolated data array of shape (n, m).
    Y_interpolated (numpy.ndarray): The interpolated labels array of shape (n,).
    """
    n = X.shape[0] - 1
    X_interpolated = np.zeros((n, X.shape[1]))
    Y_interpolated = np.zeros(n)

    # Interpolate each data point and label using simple mean interpolation
    for i in range(n):
        X_interpolated[i, :] = simple_mean_interpolation(X[i, :], X[i + 1, :])
        Y_interpolated[i] = simple_mean_interpolation(Y[i], Y[i + 1])

    return X_interpolated, Y_interpolated

def remove_dates(dates, string_list):
    """
    Removes dates from a list of dates if they are present in a given string list.

    Parameters:
    dates (list): A list of dates.
    string_list (list): A list of strings.

    Returns:
    list: A new list of dates with the dates that are present in the string list removed.
    """
    # Create a new list using list comprehension
    # Only include dates that are not present in the string list
    return [d for d in dates if d not in string_list]

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
    Given a date and path, return the training data and labels, with or without interpolation. .

    Parameters:
    - date (str): The date for which to retrieve the training data and labels.
    - path (str): The path to the directory containing the image files and the Excel file.
    - interpolation (bool): Whether to perform interpolation on the image data. Default is False.
    - time_until (str): The maximum time until which to include training data. Format: 'HHMMSS'. Default is None.

    Returns:
    - X (ndarray): The training data with shape (n_samples, n_features).
    - Y (ndarray): The labels with shape (n_samples,).

    """

    # Find all image files (V has (y, x, t) dimensions)
    V, timesDay, times, mask = load_images(date, path)

    # Filter data based on time_until parameter
    if time_until is not None and isinstance(time_until, str) and len(time_until) == 6:
        times = times[times <= time_until]
        timesDay = timesDay[timesDay <= time_until]
        V = V[:, :, :len(times)]

    times = round_seconds(times)

    if interpolation == False:
        # No interpolation
        X_T = np.zeros((mask.sum(), V.shape[2]))  # X_T is the transpose of X, X_T has (pixels, time) dimensions
        for i in range(V.shape[2]):
            X_T[:, i] = V[:, :, i][mask]
        X = X_T.T #Get images where each row is an image

        Y = np.zeros(V.shape[2])
        excel_file = f'{path}/{date}.xlsx'
        excel_data = pd.read_excel(excel_file, usecols="A,F").values  # this is a numpy array first col with timestamp format, second col with numbers
        excel_times, excel_power = return_time_and_power_from_excel_data(excel_data)
        for i, time in enumerate(times):  # times are from images
            indices = np.where(excel_times == time)[0]  # this is the index of the time in the excel file in an array, if there are multiple values, then there are time duplicates
            if indices.size > 1:
                warning_message(f"Multiple values of time {time} in the excel file.")
            #Extract correct power output for each image
            Y[i] = excel_power[indices[0]] 
    else:
        # Interpolation
        #Choose number of interpolation points between images
        n_interpolation_points_between_images = 3
        n_images = (V.shape[2] - 1) * n_interpolation_points_between_images + 1  #Number of images after interpolatino
        objects = range(V.shape[2] - 1)  #Range depending on number of original images
        dps_images = Lucas_Kanade_method(V, objects=objects) #Calculate movement
        #Do interpolation
        V_interpolation, timestamps = interpolate_flow(dps_images, V, timesDay, times, mask, objects=objects, show=False, n=n_interpolation_points_between_images)

        # Add the last image
        V_interpolation = np.concatenate((V_interpolation, np.expand_dims(V[:, :, -1], 2)), axis=2)
        timestamps.append(times[-1])

        X_T = np.zeros((mask.sum(), n_images))  # X_T is the transpose of X, X_T has (pixels, time) dimensions
        for i in range(n_images):
            X_T[:, i] = V_interpolation[:, :, i][mask]
        X = X_T.T #Get images where each row is an image

        Y = np.zeros(n_images)
        excel_file = f'{path}/{date}.xlsx'
        excel_data = pd.read_excel(excel_file, usecols="A,F").values  # this is a numpy array first col with timestamp format, second col with numbers
        excel_times, excel_power = return_time_and_power_from_excel_data(excel_data)
        for i, time in enumerate(times):  # times are from images
            indices = np.where(excel_times == time)[0]  # this is the index of the time in the excel file in an array, if there are multiple values, then there are time duplicates
            if indices.size > 1:
                warning_message(f"Multiple values of time {time} in the excel file.")
            #Extract correct power output for each image
            Y[i] = excel_power[indices[0]]
        pass

    return X, Y

def ridge_model_on_extrapolated_data(dates, test_date, alphas, target_time:str, interpolation=False):
    """
    Perform ridge model on extrapolated data.

    Args:
        dates (list): List of dates.
        test_date (str): Test date.
        alphas (list): List of regularization parameters.
        target_time (str): Target time.
        interpolation (bool, optional): Flag indicating whether to use interpolation. Defaults to False.

    Returns:
        tuple: Tuple containing errors, average absolute distances, predicted power values, true power values, and true times.
    """
    # Find the index of the test date in the list of dates
    date_index = dates.index(test_date)

    # Load data and labels from all dates before the test date
    for i in range(date_index):
        date = dates[i]
        X_temp, Y_temp = get_training_data_and_labels(date, path, interpolation=interpolation)
        print(f"Date: {date}, data shape: {X_temp.shape}, label shape: {Y_temp.shape}")
        if i == 0:
            X = X_temp
            Y = Y_temp
        else:
            X = np.vstack([X,X_temp])
            Y = np.hstack([Y,Y_temp])

    # Load data and labels before the target_time at the test date
    X_temp, Y_temp = get_training_data_and_labels(dates[date_index], path, interpolation=interpolation, time_until=target_time)
    X = np.vstack([X,X_temp])
    Y = np.hstack([Y,Y_temp])

    # Standardize the input data
    scaler = StandardScaler()
    X_standardized = scaler.fit_transform(X)

    # Train the ridge and select the model with the best regularization parameter alpha_
    model = RidgeCV(alphas=alphas, store_cv_values=True)
    model.fit(X_standardized, Y)
    print("optimal alpha is ", model.alpha_)

    # Find the optimal alpha and train again
    model = RidgeCV(model.alpha_)
    model.fit(X_standardized,Y)
    
    # Do extrapolation from the target_time of the test date, and predict the power
    V, timesDay, times, mask = load_images(test_date, path)
    print(timesDay, times)
    
    # Do extrapolation on all images after the specified target time
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
        #Do predictions
        X_test_standardized = scaler.transform(np.expand_dims(V_extrapolation[:,:,i][mask], 0)) # special purpose, the input should be a one row long 2D array
        y_prediction = model.predict(X_test_standardized)
        y_prediction_list.append(y_prediction)

        # Get the true power from the excel file
        excel_file = f'{path}/{test_date}.xlsx'
        excel_data = pd.read_excel(excel_file, usecols="A,F").values    # this is a numpy array first col with timestamp format, second col with numbers
        excel_times, excel_power = return_time_and_power_from_excel_data(excel_data)
        true_time = round_seconds([timestamps[i]])[0]   #round seconds of timestamps to find correct power output from excel fil
        indices = np.where(excel_times == true_time)[0] # this is the index of the time in the excel file in a array, if there are mutiple values,then there are time duplicates
        y_true = excel_power[indices]
        y_true_list.append(y_true)

        #Calculate MSE
        error = MSE(y_prediction, y_true)
        errors.append(error)

        #Calculate average absolute distance
        average_distance = averge_distance(y_prediction, y_true)
        average_distances.append(average_distance)

        print(f"Error at time {true_time} is {error}")
        print(f"Average absolute distance at time {true_time} is {average_distance}")
        true_time_list.append(true_time)
        
        # Append new image to the training data as this data should be used for future predictions.
        if i != V_extrapolation.shape[2] - 1:    # prevent overflow
            X, Y = np.vstack([X, np.expand_dims(V[:,:,time_index + i + 1][mask], axis=0)]), np.hstack([Y, y_true])
            X_standardized = scaler.fit_transform(X)
            model.fit(X_standardized,Y)
    
    return errors, average_distances, y_prediction_list, y_true_list, true_time_list

def MSE(Y_pred, Y_true):
    """
    Calculates the mean squared error (MSE) between predicted and true values.

    Parameters:
    - Y_pred (array-like): Predicted values.
    - Y_true (array-like): True values.

    Returns:
    - mse (float): Mean squared error.

    """
    # Calculate the squared difference between predicted and true values
    squared_diff = (Y_pred - Y_true) ** 2

    # Calculate the mean squared error
    mse = np.mean(squared_diff)

    return mse

def averge_distance(Y_pred, Y_true):
    """
    Calculate the average distance between predicted values and true values.

    Parameters:
    - Y_pred (array-like): Predicted values.
    - Y_true (array-like): True values.

    Returns:
    - float: Average distance between predicted values and true values.
    """
    return np.mean(abs((Y_pred - Y_true)))

def model_performance_test(dates):
    # This function evaluates the performance of different ridge regression models on the given dates.
    # It calculates and plots the residuals for each model.

    # Define the range of alpha values for RidgeCV
    alphas = np.logspace(-6, 6, 200)

    ################Residuals for model without interpolation#################

    # Load the input data and labels
    X = np.load("X.npy")
    Y = np.load("Y.npy")

    # Standardize the input data
    scaler = StandardScaler()
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=2)
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Create the RidgeCV model
    m = RidgeCV(alphas=alphas)

    # Fit the model to the training data
    m.fit(X_train, Y_train)

    # Get the optimal alpha value
    optimal_alpha = m.alpha_

    # Create and fit the Ridge regression model with the optimal alpha
    model = Ridge(alpha=optimal_alpha)
    model.fit(X_train, Y_train)

    # Predict the values for the test data
    Y_prediction = model.predict(X_test)

    # Calculate the residuals
    residuals = Y_prediction - Y_test

    # Print the mean squared error (MSE) of the residuals
    print(np.mean(residuals**2))

    # Plot the residuals
    plt.scatter(Y_prediction, residuals)
    plt.xlabel('Predicted values')
    plt.ylabel('Residuals')
    plt.axhline(y=0, color='r', linestyle='--')
    plt.title("Residuals for model without interpolation")
    plt.savefig("Residuals_model_without_interpolation.svg")
    plt.clf()

    ##############Residuals for model with interpolation"###############

    # Load the input data and labels
    X = np.load("X.npy")
    Y = np.load("Y.npy")

    # Standardize the input data
    scaler = StandardScaler()
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=2)

    # Get the interpolated data and labels
    X_interpolated, Y_interpolated = get_interpolated_data_and_labels(X, Y)

    # Standardize the training data with the interpolated data
    X_train = scaler.fit_transform(np.vstack((X_train, X_interpolated)))
    Y_train = np.hstack((Y_train, Y_interpolated))

    # Standardize the test data
    X_test = scaler.transform(X_test)

    # Create the RidgeCV model
    m = RidgeCV(alphas=alphas)

    # Fit the model to the training data
    m.fit(X_train, Y_train)

    # Get the optimal alpha value
    optimal_alpha = m.alpha_

    # Create and fit the Ridge regression model with the optimal alpha
    model = Ridge(alpha=optimal_alpha)
    model.fit(X_train, Y_train)

    # Predict the values for the test data
    Y_prediction = model.predict(X_test)

    # Calculate the residuals
    residuals = Y_prediction - Y_test

    # Print the mean squared error (MSE) of the residuals
    print(np.mean(residuals**2))

    # Plot the residuals
    plt.scatter(Y_prediction, residuals)
    plt.xlabel('Predicted values')
    plt.ylabel('Residuals')
    plt.axhline(y=0, color='r', linestyle='--')
    plt.title("Residuals for model with simple weighted interpolation")
    plt.savefig("Residuals_model_with_simple_weighted_interpolation.svg")
    plt.clf()

    ##############Residuals model with optical flow interpolation#############

    # Load the input data and labels
    X = np.load("X.npy")
    Y = np.load("Y.npy")

    # Standardize the input data
    scaler = StandardScaler()

    # Initialize empty arrays for X and Y
    X = np.array([])
    Y = np.array([])

    # Loop through each date and append the training data and labels
    for i, date in enumerate(dates):
        X_temp, Y_temp = get_training_data_and_labels(date, path, interpolation=True)
        print(f"Date: {date}, data shape: {X_temp.shape}, label shape: {Y_temp.shape}")
        if i == 0:
            X = X_temp
            Y = Y_temp
        else:
            X = np.vstack([X,X_temp])
            Y = np.hstack([Y,Y_temp])

    # Split the data into training and test sets
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=2)

    # Standardize the training data
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Create the RidgeCV model
    m = RidgeCV(alphas=alphas)

    # Fit the model to the training data
    m.fit(X_train, Y_train)

    # Get the optimal alpha value
    optimal_alpha = m.alpha_

    # Create and fit the Ridge regression model with the optimal alpha
    model = Ridge(alpha=optimal_alpha)
    model.fit(X_train, Y_train)

    # Predict the values for the test data
    Y_prediction = model.predict(X_test)

    # Calculate the residuals
    residuals = Y_prediction - Y_test

    # Print the mean squared error (MSE) of the residuals
    print(np.mean(residuals**2))

    # Plot the residuals
    plt.scatter(Y_prediction, residuals)
    plt.xlabel('Predicted values')
    plt.ylabel('Residuals')
    plt.axhline(y=0, color='r', linestyle='--')
    plt.title("Residuals model with optical flow interpolation")
    plt.savefig("Residuals_model_with_optical_flow_interpolation.svg")

    return

if __name__ == '__main__':
    # If this is the main file being run then do the extrapolation and prediction with and without interpolation and save results. 
    path = 'Project4/Processedfull'
    files_in_directory = os.listdir(path)
    dates = [file.replace('.xlsx','') for file in files_in_directory if file.endswith('.xlsx')]
    sort_dates(dates)
    
    # model_performance_test(dates)
   
    time_until = "063000"
    number_of_alphas = 1000
    alpha_vals = np.logspace(-6, 6, number_of_alphas)
    test_date= '20240306'
    errors, average_distances, y_prediction_list, y_true_list, true_time_list = ridge_model_on_extrapolated_data(dates, test_date, alpha_vals, time_until)
    errors_interpolation, average_distances_interpolation, y_prediction_list_interpolation, _, _ = ridge_model_on_extrapolated_data(dates, test_date, alpha_vals, time_until, interpolation=True)
    np.save('errors.npy', errors)
    np.save('average_distances.npy', average_distances)
    np.save('y_prediction_list.npy', y_prediction_list)
    np.save('y_true_list.npy', y_true_list)
    np.save('true_time_list.npy', true_time_list)
    np.save('errors_interpolation.npy', errors_interpolation)
    np.save('average_distances_interpolation.npy', average_distances_interpolation)
    np.save('y_prediction_list_interpolation.npy', y_prediction_list_interpolation)   
    pass
        

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