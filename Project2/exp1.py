import os
import helpFunctions as hf 
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.pyplot import imread 
from scipy.stats import norm

dirIn = rf'{os.getcwd()}/Project2/data/'  

# Load multi spectral image and annotations from day 1
multiIm, annotationIm = hf.loadMulti('multispectral_day01.mat' , 'annotation_day01.png', dirIn)

# Extract fat and meat pixel values
[fatPix, fatR, fatC] = hf.getPix(multiIm, annotationIm[:,:,1])
[meatPix, meatR, meatC] = hf.getPix(multiIm, annotationIm[:,:,2])

# Calculate thresholds
t = np.mean(meatPix,0) + (np.mean(fatPix,0)-np.mean(meatPix,0)) / 2

# What about the background? If we do it this way i guess all background pixels will be classified as meat
# since the mean of the background pixels is in between fat and meat and closest to meat?

# Create binary image
binIm = np.zeros(19)
for i in range(19):
    binIm[i] = multiIm[:,:,i] > t   # meat pixels are black, fat are white

