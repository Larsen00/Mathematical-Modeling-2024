# -*- coding: utf-8 -*-
"""
Created on Tue Feb  6 10:29:09 2024

@author: jonas
"""

import glob
import skimage
import matplotlib.pyplot as plt
import numpy as np
import scipy
import random
import os

# Get the current working directory
working_dir = os.getcwd()

#%%
## Problem 1 - Loading and displaying a toy problem:
    
# Define the path to the .png files in the 'toyProblem_F22' subdirectory
pngs = glob.glob(working_dir + '/toyProblem_F22/*.png')
assert len(pngs) == 64

#%%
# make image list
images = [skimage.color.rgb2gray(plt.imread(i)) for i in pngs]

# make image array
V = np.dstack(images)
V.shape


#%%
# Problem 2.1

# Compute the gradient of each image in the x-direction (horizontal difference)
Vx = [img[:,1:] - img[:,0:-1] for img in images]

# Compute the gradient of each image in the y-direction (vertical difference)
Vy = [img[1:,:] - img[0:-1,:] for img in images]

# Compute the temporal gradient of each image (difference between consecutive images)
Vt = [img1-img2 for img1, img2 in zip(images[1:], images[0:-1])]

# Compute the gradient in x, y and t direction
Vx = V[1:, :, :] - V[0:-1, :, :]
Vy = V[:, 1:, :] - V[:, 0:-1, :]
Vt = V[:, :, 1:] - V[:, :, 0:-1]
Vx.shape, Vy.shape, Vt.shape
