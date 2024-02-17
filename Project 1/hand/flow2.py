# Please run flow1.py then run flow2.py

import glob
import skimage
import matplotlib.pyplot as plt
import numpy as np

step_size = 5

# path = input("Enter the path of the folder containing the images. This will be given to the \"glob\" function.")
path = r"."
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

print("Filtering out noises...")
# plot with filtering out noises
dps_images = np.load("flow.npy")
dps_images_filtered = dps_images.copy()
for x in range(0,V.shape[0],step_size):
  for y in range(0,V.shape[1],step_size):
        for t in range(V.shape[2]):
            ## filter out arrows with small magnitude (<= 5)
            if np.linalg.norm(dps_images_filtered[:,x,y,t]) <= 40 or np.linalg.norm(dps_images_filtered[:,x,y,t]) >= 60:
                dps_images_filtered[:,x,y,t] = np.zeros(2)
np.save("flow_filtered.npy", dps_images_filtered)
print("Filtering out noises done!")

print("Plotting...")
dps_images_filtered = np.load("flow_filtered.npy")
X, Y = np.meshgrid(np.arange(0, V.shape[0], step_size), np.arange(0, V.shape[1], step_size))
for t in range(V.shape[2]):
    plt.imshow(ims_with_color[t])
    plt.quiver(Y, X, dps_images_filtered[1,X,Y,t], dps_images_filtered[0,X,Y,t], scale=700, width=.003)
    # plt.pause(1)
    plt.savefig(f"./hand_with_quivers/flow-{t:04d}.png")
    plt.clf()
print("Plotting done!")