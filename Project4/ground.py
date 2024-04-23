import openpyxl
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split


def extract_groundintensity ():
    folder_path = 'Project4/processedfull'
    files_in_directory = os.listdir(folder_path)
    dates = [file.replace('.xlsx','') for file in files_in_directory if file.endswith('.xlsx')]

    # Find all image files
    file_name = []
    for day in dates:
        file_name += glob.glob(f'{folder_path}/{day[6:8]}/*natural_color.npy')

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
    return x


