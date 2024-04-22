# Please run flow1.py then run flow2.py

import glob
import matplotlib.pyplot as plt
import numpy as np
import os.path

path = "Project4"

if not os.path.isdir(f'{path}/_static'):
    os.mkdir(f"{path}/_static")

stride = 2

# dates of images
dates = ['0317', '0318', '0319', '0326', '0329', '0331']

# Find all image files
file_name = []
for day in dates:
    file_name += glob.glob(f'{path}/processed/{day[2:4]}/*natural_color.npy')

# Load binary mask outlining Denmark
mask = np.load(f'{path}/processed/mask.npy')

# Allocate memory and load image data
times = []
timesDay = []
ims = []
for i, entry in enumerate(file_name):
    img = np.load(entry)
    
    dummy = img[:,:,0]+img[:,:,1]-2*img[:,:,2]
    dummy = dummy*mask
    dummy[dummy < 0.0] = 0.0
    
    # Find time information in filename
    ind = entry.find('202403')
    
    times.append(entry[ind+8:ind+14])   # gives time of day hhmmss
    timesDay.append(entry[ind+6:ind+8]) # gives the date
    
    print(f"Progress: {(i+1)/len(file_name)*100:.2f}%", end="\r")

    # read images
    ims.append(dummy)
    
    
timesDay = np.array(timesDay)
times = np.array(times)

# make image array (y,x,t)
V = np.dstack(ims)
print(f"Reading images done! {len(ims)} images read.")

# answer = input("Want to create new flow_filtered.npy? (Might take some time) [y/N]")
# if (answer == "yes") | (answer == "y"):
print("Filtering out noises and plot...")
# plot with filtering out noises
dps_images = np.load(f"{path}/flow.npy")
dps_images_filtered = dps_images.copy()
# for t in range(V.shape[2]):
for t in range(10):
    plt.imshow(ims[t], cmap="gray")
    for y in range(0,V.shape[0],stride):
        for x in range(0,V.shape[1],stride):
            ## filter out arrows with small magnitude (<= 5)
            tmp = np.linalg.norm(dps_images_filtered[:,y,x,t])
            if tmp > 2 and mask[y,x] == True:
                plt.quiver(x, y, dps_images_filtered[0,y,x,t], dps_images_filtered[1,y,x,t], color="red", width=.003)
    plt.tight_layout()
    plt.title(f"Optical Flow Filtered 202403{timesDay[t]}_{times[t]}")
    plt.savefig(f"{path}/_static/flow_filtered_202403{timesDay[t]}_{times[t]}.png")
    plt.clf()
    print(f"Plotting image {t+1}/{V.shape[2]}.", end="\r")
print("All Done! Program terminates.")