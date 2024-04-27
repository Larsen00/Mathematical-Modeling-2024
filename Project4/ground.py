import glob
import numpy as np
import os
import matplotlib.pyplot as plt

# lav en kernel filter på ground før den retuneres

def extract_groundintensity ():
    folder_path = 'Project4/Processedfull'
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
        Xdata[:,i] = dummy[mask].flatten()
        
        # It is assumed that the maximum value found of a pixel is the gound without a cloud
        groundIntensity = np.maximum(groundIntensity, Xdata[:,i])
        
        # Find time information in filename
        ind = entry.find('202403')
        
        
        times.append(entry[ind+8:ind+14])   # gices time of day hhmmss
        timesDay.append(entry[ind+6:ind+8]) # gives the date
        
        i +=1

    x = np.full(mask.shape, np.nan)
    x[mask] = groundIntensity
    return x

if __name__ == "__main__":
    x = extract_groundintensity()
    plt.imshow(x, cmap='viridis')
    plt.show()