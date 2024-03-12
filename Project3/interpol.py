import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from math import radians, sin, cos, sqrt, atan2
import scipy as sp

# Function to calculate Haversine distance
def haversine_distance(lat1, lon1, lat2, lon2):
    R = 6373.0  # Earth radius in kilometers

    dlat = radians(lat2 - lat1)
    dlon = radians(lon2 - lon1)

    a = sin(dlat / 2) ** 2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlon / 2) ** 2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))

    distance = R * c
    return distance * 1000  # Convert to meters



df =pd.read_csv('channel_data.txt',delimiter= ',')
distances = np.array([0])
for i in range(len(df)-1):
    distances = np.append(distances,distances[-1]+haversine_distance(df['Latitude'][i],df['Longtitude'][i],df['Latitude'][i+1],df['Longtitude'][i+1]))
height = df['height']
f = sp.interpolate.interp1d(distances,height)
xs = np.arange(0,79000,250) # we only want an x value every 250 metres
interpol_heights = f(xs)
np.savetxt('interpol_heights',interpol_heights,delimiter=',')
