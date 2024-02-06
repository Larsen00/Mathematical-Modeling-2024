# -*- coding: utf-8 -*-
"""
Created on Tue Jan 30 09:51:11 2024

@author: jonas
"""

import glob
from skimage import io, color
import numpy as np
import time 
import scipy
import matplotlib.pyplot as plt
import os
#%%

# impotere
working_dir = os.getcwd() 
print(working_dir)
#%%

folder = "C:/Users/jonas/OneDrive - Danmarks Tekniske Universitet/DTU/Matematisk Modelering/Opgaver/"
paths = glob.glob(working_dir + 'toyProblem_F22/*.png')
images = [color.rgb2gray(io.imread(path)) for path in paths]

#%%
for img in images:
    io.imshow(img)
    io.show()
    time.sleep(1)

#%%
# Problem 2.1

# Gradient
Vx = [img[:,1:] - img[:,0:-1] for img in images]
Vy = [img[1:,:] - img[0:-1,:] for img in images]
Vt = [img1-img2 for img1, img2 in zip(images[1:], images[0:-1])]

#%%
#Problem 2.2

# avage Vx and Vt
avage_gradient = [(img1[:-1,:]+img2[:,:-1]) for img1, img2 in zip(Vx, Vy)]

#%%
#Kernel
sobel_kernel = [scipy.ndimage.sobel(img) for img in images]
prewitt_kernel = [scipy.ndimage.prewitt(img) for img in images]

for i in range(0,len(Vt)):
    plt.figure(figsize=(10, 10))  # Adjust the figure size as needed
    plt.subplot(3, 3, 1)
    plt.imshow(images[i],cmap='gray')
    plt.title(f"Frame {i} - Original Image")
    
    plt.subplot(3, 3, 2)
    plt.imshow(Vx[i],cmap='gray')
    plt.title('dx')
    
    plt.subplot(3, 3, 3)
    plt.imshow(Vy[i],cmap='gray')
    plt.title('dy')
    
    plt.subplot(3, 3, 4)
    plt.imshow(Vt[i],cmap='gray')
    plt.title('dt')
    
    plt.subplot(3,3,5)
    plt.imshow(sobel_kernel[i], cmap='gray')
    plt.title('sobel')
    
    plt.subplot(3,3,6)
    plt.imshow(prewitt_kernel[i], cmap='gray')
    plt.title('prewitt')
    
    plt.subplot(3,3,7)
    plt.imshow(avage_gradient[i], cmap='gray')
    plt.title('both')
    
    plt.show()

