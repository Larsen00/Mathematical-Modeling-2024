# Please run flow1.py then run flow2.py

import glob
import skimage
import matplotlib.pyplot as plt
import numpy as np
import scipy

# path = input("Enter the path of the folder containing the images. This will be given to the \"glob\" function.")
path = r".\toilet-paper"
pngs = glob.glob(path + "/*.png")

# make image list
ims_with_color = []
ims = []
print("Reading images...")
for status, i in enumerate(pngs):
    # read in image in grayscale
    ims_with_color.append(plt.imread(i))
    # read in image in grayscale
    ims.append(skimage.color.rgb2gray(plt.imread(i)))
    print(f"Progress: {(status+1)/len(pngs)*100:.2f}%", end="\r")

# make image array
V = np.dstack(ims)
print(f"Reading images done! {len(ims)} images read.")

# Gaussian filter
print("Creating Gaussian filter...")
# sigma = int(input("Enter the sigma value for the Gaussian filter: "))
sigma = 1
gaussian_V = scipy.ndimage.gaussian_filter(V, sigma, order=1)
Vx_gaussian = scipy.ndimage.gaussian_filter(V, sigma, order=1, axes=1)
Vy_gaussian = scipy.ndimage.gaussian_filter(V, sigma, order=1, axes=0)
Vt_gaussian = scipy.ndimage.gaussian_filter(V, sigma, order=1, axes=2)
print("Gaussian filter done!")

print("Calculating optical flow...")
N = 5
n = int((N-1)//2)
step_size = 3
t = 1
print("\tPreallocating memory...")
dps_images = np.empty((2, V.shape[0], V.shape[1], V.shape[2]))
print("\tPreallocating memory done!")
total_volume = V.shape[0]*V.shape[1]*V.shape[2]
for t in range(V.shape[2]):
    im = ims[t]
    dps_image = np.empty((2, V.shape[0], V.shape[1]))
    for x in range(0,V.shape[0],step_size):
        for y in range(0,V.shape[1],step_size):
            x_lower, x_upper = np.maximum(0, x-n), np.minimum(im.shape[0], x+n+1)
            y_lower, y_upper = np.maximum(0, y-n), np.minimum(im.shape[1], y+n+1)
            Vx_col = Vx_gaussian[x_lower:x_upper, y_lower:y_upper, t].reshape(-1,1)
            Vy_col = Vy_gaussian[x_lower:x_upper, y_lower:y_upper, t].reshape(-1,1)
            A = np.hstack([Vx_col, Vy_col])
            b = -Vt_gaussian[x_lower:x_upper, y_lower:y_upper, t].reshape(-1,1)
            dp = np.linalg.lstsq(A, b, rcond=None)[0]
            dps_image[:,x,y] = dp[:,0]
            print(f"Progress: {(t*V.shape[0]*V.shape[1] + x*V.shape[1] + y)/total_volume*100:.2f}%", end="\r")
    dps_images[:,:,:,t] = dps_image
np.save("flow.npy", dps_images)
print("All Done! Program terminates.")