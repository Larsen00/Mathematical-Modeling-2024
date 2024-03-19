import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from math import radians, sin, cos, sqrt, atan2
import scipy as sp
import os

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
for i in range(len(df)-1):
    distances = np.append(distances,distances[-1]+haversine_distance(df['Latitude'][i],df['Longtitude'][i],df['Latitude'][i+1],df['Longtitude'][i+1]))
height = df['height']
f = sp.interpolate.interp1d(distances,height)
xs = np.arange(0,79000,250) # we only want an x value every 250 metres

#Get data for X and R
bombs_1 = np.array([int(i) for i in np.round(pd.read_csv('Project3/res/p2_X.txt', header=None).to_numpy()).flatten()]).astype(bool)
dirt_1 = np.round(pd.read_csv('Project3/res/p2_R.txt', header=None).to_numpy()).flatten()
bombs_2 = np.array([int(i) for i in np.round(pd.read_csv('Project3/res/p4_X.txt', header=None).to_numpy()).flatten()]).astype(bool)
dirt_2 = np.round(pd.read_csv('Project3/res/p4_R.txt', header=None).to_numpy()).flatten()
heights = f(xs) #just the heights at a 250m interval

bomb_mask = xs[bombs_1]
interpol_heights = f(xs)
np.savetxt('Project3/interpol_heights.txt',interpol_heights,delimiter=',')
plt.plot(xs,f(xs))
plt.scatter(bomb_mask,f(bomb_mask),color = 'red',s=10, label='Bombs')

plt.xlabel('Distance to sea')
plt.ylabel('Interpolated height above sea-level')
plt.legend()
plt.savefig(f'{os.getcwd()}/Project3/plots/heights_with_bomb_location.png')
print("Saved")
plt.clf()

# New plots looking at depth
depths1 = heights-dirt_1
depths2 = heights-dirt_2
plt.plot(xs,depths1)
plt.plot(xs,depths2)
plt.savefig(f'{os.getcwd()}/Project3/plots/depth_compaison.png')