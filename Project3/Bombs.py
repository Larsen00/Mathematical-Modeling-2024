import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from math import radians, sin, cos, sqrt, atan2
import scipy as sp
import os
from haversine import haversine
import ast

df =pd.read_csv('Project3/channel_data.txt',delimiter= ',')
distances = np.array([0])
distances_old = np.array([0])
for i in range(len(df)-1):
    distances = np.append(distances,distances[-1]+haversine((df['Latitude'][i],df['Longtitude'][i]),(df['Latitude'][i+1],df['Longtitude'][i+1])))
distances *= 1000
height = df['height']
f = sp.interpolate.interp1d(distances,height)
xs = np.arange(0,79000,250) # we only want an x value every 250 metres

#Get data for X and R
df1 = pd.read_csv('Project3/res/p3_X.txt', header=None)[0].apply(lambda x: ast.literal_eval(x)[0] if x else None) #fix the format
df2 = pd.read_csv('Project3/res/p4_X.txt', header=None)[0].apply(lambda x: ast.literal_eval(x)[0] if x else None) #fix the format
df3 =  pd.read_csv('Project3/res/p6_X.txt', header=None,sep = 'n')
df3.columns = ['bombs']
df3['bombs'] = df3['bombs'].apply(lambda x: ast.literal_eval(x)[0:3] if x else None)

bombs_1 = np.array([int(i) for i in np.round(df1).to_numpy().flatten()]).astype(bool)
bombs_2 = np.array([int(i) for i in np.round(df2).to_numpy().flatten()]).astype(bool)
bombs_3 = pd.DataFrame(df3['bombs'].tolist(), columns=['Bomb1', 'Bomb2', 'Bomb3']).applymap(lambda x: bool(round(float(x))))


dirt_1 = np.round(pd.read_csv('Project3/res/p3_R.txt', header=None).to_numpy()).flatten()
dirt_2 = np.round(pd.read_csv('Project3/res/p4_R.txt', header=None).to_numpy()).flatten()
dirt_3 = np.round(pd.read_csv('Project3/res/p6_R.txt', header=None).to_numpy()).flatten()

heights = f(xs) #just the heights at a 250m interval
#Get bomb location and the corresponding depths
bomb_mask_1 = xs[bombs_1]
bomb_mask_2 = xs[bombs_2]

bomb_mask_3_1 = xs[bombs_3['Bomb1']]
bomb_mask_3_2 = xs[bombs_3['Bomb2']]
bomb_mask_3_3 = xs[bombs_3['Bomb3']]

min_depth = np.linspace(-10,-10,len(xs))
depths1 = heights-dirt_1
depths2 = heights-dirt_2
depths3 = heights-dirt_3

dirt_1_masked = depths1[bombs_1]
dirt_2_masked = depths2[bombs_2]

dirt_3_1_masked = depths3[bombs_3['Bomb1']]
dirt_3_2_masked = depths3[bombs_3['Bomb2']]
dirt_3_3_masked = depths3[bombs_3['Bomb3']]
#Plot 1
#Plot height and bomb placement
interpol_heights = f(xs)
np.savetxt('Project3/interpol_heights.txt',interpol_heights,delimiter=',')
plt.plot(xs,f(xs))
plt.scatter(bomb_mask_1,f(bomb_mask_1),color = 'green',s=10, label='Bombs')
plt.title("Bomb placement with obj func. 1 and interpolated heights")
plt.xlabel('Distance to sea (m)')
plt.ylabel('Interpolated height above sea-level (m)')
plt.legend()
plt.savefig(f'{os.getcwd()}/Project3/plots/heights_with_bomb_location.svg')

plt.clf()

# New plots looking at depth
#First get line of min depth
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))  # 1 row, 2 columns, optional figure size

# First subplot
ax1.plot(xs,depths1,label = 'Depth with 1. obj. function')
ax1.plot(xs, min_depth, label='min_depth', linestyle='--')  
ax1.scatter(bomb_mask_1,dirt_1_masked,color = 'green',s=10, label='Bombs with 1. obj. function')
ax1.set_ylim([min(depths1)-10, 10])
ax1.set_xlabel('Distance to sea (m)')
ax1.set_ylabel('Depth of channel (m)')
ax1.set_title("Bomb placement and depths with obj. function 1")
ax1.legend()
ax1.text(0.01, 0.995, f'Bombs used: {len(bomb_mask_1)}', transform=ax1.transAxes, verticalalignment='top')
ax1.text(0.01, 0.97, f'Dirt removed: {sum(dirt_1)}', transform=ax1.transAxes, verticalalignment='top')


# Second subplot
ax2.plot(xs, depths2, label='Depth with 2. objective')
ax2.set_title("Bomb placement and depths with obj. function 2")
ax2.plot(xs, min_depth, label='min_depth', linestyle='--')  # Duplicating for the second objective
ax2.scatter(bomb_mask_2,dirt_2_masked,color = 'green',s=10, label='Bombs with 2. objective')
ax2.set_ylim([min(depths2)-10, 10])
ax2.set_xlabel('Distance to sea (m)')
# ax2.set_ylabel('Depth of channel')  # Optional, as it's the same as the first subplot
ax2.legend()
ax2.text(0.01, 0.995, f'Bombs used: {len(bomb_mask_2)}', transform=ax2.transAxes, verticalalignment='top')
ax2.text(0.01, 0.97, f'Dirt removed: {sum(dirt_2)}', transform=ax2.transAxes, verticalalignment='top')

plt.tight_layout()  # Adjust subplots to fit into the figure area.
plt.savefig(f'{os.getcwd()}/Project3/plots/depth_comparison.svg')
plt.clf()

fig, (ax1) = plt.subplots(1, 1, figsize=(12, 6))  # 1 row, 1 columns, optional figure size
# Second subplot
ax1.plot(xs, depths3, label='Depth with 2. objective and 3 bomb types')
ax1.set_title("Bomb placement and depths with obj. function 2 and 3 bomb types")

ax1.plot(xs, min_depth, label='min_depth', linestyle='--')  # Duplicating for the second objective
ax1.scatter(bomb_mask_3_1,dirt_3_1_masked,color = 'green',s=10, label='Bomb 1')
ax1.scatter(bomb_mask_3_2,dirt_3_2_masked,color = 'red',s=10, label='Bomb 2')
ax1.scatter(bomb_mask_3_3,dirt_3_3_masked,color = 'black',s=10, label='Bomb 3')

ax1.set_ylim([min(depths3)-10, 10])

ax1.set_xlabel('Distance to sea (m)')
ax1.set_ylabel('Depth of channel (m)')
# ax2.set_ylabel('Depth of channel')  # Optional, as it's the same as the first subplot
ax1.legend()
ax1.text(0.01, 0.995, f'Bombs used: {len(bomb_mask_3_1)+len(bomb_mask_3_2)+len(bomb_mask_3_3)}', transform=ax1.transAxes, verticalalignment='top')
ax1.text(0.01, 0.97, f'Dirt removed: {sum(dirt_3)}', transform=ax1.transAxes, verticalalignment='top')

plt.tight_layout()  # Adjust subplots to fit into the figure area.
plt.savefig(f'{os.getcwd()}/Project3/plots/3BombTypes.svg')
plt.clf()

print("Saved")
