import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from math import radians, sin, cos, sqrt, atan2
import scipy as sp
import os
from haversine import haversine

# Function to calculate Haversine distance
def haversine_distance(lat1, lon1, lat2, lon2):
    R = 6373.0  # Earth radius in kilometers

    dlat = radians(lat2 - lat1)
    dlon = radians(lon2 - lon1)

    a = sin(dlat / 2) ** 2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlon / 2) ** 2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))

    distance = R * c
    return distance * 1000  # Convert to meters

df =pd.read_csv('Project3/channel_data.txt',delimiter= ',')
distances = np.array([0])
distances_old = np.array([0])
for i in range(len(df)-1):
    distances = np.append(distances,distances[-1]+haversine((df['Latitude'][i],df['Longtitude'][i]),(df['Latitude'][i+1],df['Longtitude'][i+1])))
    #distances_old = np.append(distances,distances[-1]+haversine_distance(df['Latitude'][i],df['Longtitude'][i],df['Latitude'][i+1],df['Longtitude'][i+1]))
distances *= 1000
height = df['height']
f = sp.interpolate.interp1d(distances,height)
xs = np.arange(0,79000,250) # we only want an x value every 250 metres

#Get data for X and R
bombs_1 = np.array([int(i) for i in np.round(pd.read_csv('Project3/res/p2_X.txt', header=None).to_numpy()).flatten()]).astype(bool)
dirt_1 = np.round(pd.read_csv('Project3/res/p2_R.txt', header=None).to_numpy()).flatten()
bombs_2 = np.array([int(i) for i in np.round(pd.read_csv('Project3/res/p4_X.txt', header=None).to_numpy()).flatten()]).astype(bool)
dirt_2 = np.round(pd.read_csv('Project3/res/p4_R.txt', header=None).to_numpy()).flatten()
heights = f(xs) #just the heights at a 250m interval
#Get bomb location and the corresponding depths
bomb_mask_1 = xs[bombs_1]
bomb_mask_2 = xs[bombs_2]

min_depth = np.linspace(-10,-10,len(xs))
depths1 = heights-dirt_1
depths2 = heights-dirt_2

dirt_1_masked = depths1[bombs_1]
dirt_2_masked = depths2[bombs_2]
#Plot 1
#Plot height and bomb placement
interpol_heights = f(xs)
np.savetxt('Project3/interpol_heights.txt',interpol_heights,delimiter=',')
plt.plot(xs,f(xs))
plt.scatter(bomb_mask_1,f(bomb_mask_1),color = 'red',s=10, label='Bombs')

plt.xlabel('Distance to sea')
plt.ylabel('Interpolated height above sea-level')
plt.legend()
plt.savefig(f'{os.getcwd()}/Project3/plots/heights_with_bomb_location.png')

plt.clf()

# New plots looking at depth
#First get line of min depth
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))  # 1 row, 2 columns, optional figure size

# First subplot
ax1.plot(xs,depths1,label = 'Depth with 1. objective')
ax1.plot(xs, min_depth, label='min_depth', linestyle='--')  # Assuming you want this on both plots
ax1.scatter(bomb_mask_1,dirt_1_masked,color = 'red',s=10, label='Bombs with 1. objective')
ax1.set_xlabel('Distance to sea')
ax1.set_ylabel('Depth of channel')
ax1.legend()
ax1.text(0.01, 0.95, f'Bombs used: {len(bomb_mask_1)}', transform=ax1.transAxes, verticalalignment='top')


# Second subplot
ax2.plot(xs, depths2, label='Depth with 2. objective')
ax2.plot(xs, min_depth, label='min_depth', linestyle='--')  # Duplicating for the second objective
ax2.scatter(bomb_mask_2,dirt_2_masked,color = 'green',s=10, label='Bombs with 2. objective')
ax2.set_xlabel('Distance to sea')
# ax2.set_ylabel('Depth of channel')  # Optional, as it's the same as the first subplot
ax2.legend()
ax2.text(0.01, 0.95, f'Bombs used: {len(bomb_mask_2)}', transform=ax2.transAxes, verticalalignment='top')

plt.tight_layout()  # Adjust subplots to fit into the figure area.
plt.savefig(f'{os.getcwd()}/Project3/plots/depth_comparison.png')
plt.clf()


print("Saved")
# plt.plot(xs,depths1,label = 'Depth with 1. objective')
# plt.plot(xs,depths2,label = 'Depth with 2. objective')
# plt.plot(xs,min_depth, label = 'min_depth')
# plt.scatter(bomb_mask_1,dirt_1_masked,color = 'red',s=10, label='Bombs with 1. objective')
# plt.scatter(bomb_mask_2,dirt_2_masked,color = 'yellow',s=10, label='Bombs with 2. objective')
# plt.xlabel('Distance to sea')
# plt.ylabel('Depth of channel')
# plt.legend()
# plt.savefig(f'{os.getcwd()}/Project3/plots/depth_comparison.png')