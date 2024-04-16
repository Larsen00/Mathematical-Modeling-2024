# It is clear that the lightness will affect optical flow a lot
import glob
import numpy as np
import scipy.ndimage

stride = 1

# dates of images
dates = ['0317', '0318', '0319', '0326', '0329', '0331']

path = "Project4"

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
    
    # read images
    ims.append(dummy)

    # NOTE Does it make sense to make optical between days? dates.index(entry[ind+6:ind+8])
    print(f"Progress: {(i+1)/len(file_name)*100:.2f}%", end="\r")
  
timesDay = np.array(timesDay)
times = np.array(times)

# make image array (y,x,t)
V = np.dstack(ims)
print(f"Reading images done! {len(ims)} images read.")

# Gaussian filter
# sigma = int(input("Enter the sigma value for the Gaussian filter: "))
sigma = 1
print(f"Creating Gaussian filter with sigma = {sigma}...")
# gaussian_V = scipy.ndimage.gaussian_filter(V, sigma, order=1)
Vy_gaussian = scipy.ndimage.gaussian_filter(V, sigma, order=1, axes=0)
Vx_gaussian = scipy.ndimage.gaussian_filter(V, sigma, order=1, axes=1)
Vt_gaussian = scipy.ndimage.gaussian_filter(V, sigma, order=1, axes=2)
print("Gaussian filter done!")

print("Calculating optical flow...")
n = 3
print(f"\tNeighbourhood size: {n}")
print("\tPreallocating memory...")
dps_images = np.zeros((2, V.shape[0], V.shape[1], V.shape[2]))
print("\tPreallocating memory done!")
total_volume = V.shape[0]*V.shape[1]*V.shape[2]
for t in range(V.shape[2]):
    dps_image = np.zeros((2, V.shape[0], V.shape[1]))
    for y in range(0,V.shape[0],stride):
        for x in range(0,V.shape[1],stride):
            x_lower, x_upper = np.maximum(0, x-n), np.minimum(V.shape[1], x+n+1)
            y_lower, y_upper = np.maximum(0, y-n), np.minimum(V.shape[0], y+n+1)
            Vx_col = Vx_gaussian[y_lower:y_upper, x_lower:x_upper, t].reshape(-1,1)
            Vy_col = Vy_gaussian[y_lower:y_upper, x_lower:x_upper, t].reshape(-1,1)
            A = np.hstack([Vx_col, Vy_col])
            b = -Vt_gaussian[y_lower:y_upper, x_lower:x_upper, t].reshape(-1,1)
            dp = np.linalg.lstsq(A, b, rcond=None)[0]
            dps_image[:,y,x] = dp[:,0]
            print(f"Progress: {(t*V.shape[0]*V.shape[1] + y*V.shape[1] + x)/total_volume*100:.2f}%", end="\r")
    dps_images[:,:,:,t] = dps_image
np.save(f"{path}/flow.npy", dps_images)
print("All Done! Program terminates.")