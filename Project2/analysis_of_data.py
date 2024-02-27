import os

import helpFunctions as hf 
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.pyplot import imread 
from scipy.stats import norm

dirIn = '/Users/estherholstoeksnebjerg/Desktop/02526 Mathematical Modeling/Mathematical-Modeling-2024/Project 2/data/'

# Load multi spectral image and annotations from day 1
multiIm, annotationIm = hf.loadMulti('multispectral_day01.mat' , 'annotation_day01.png', dirIn)

# Extract fat and meat pixel values
[fatPix, fatR, fatC] = hf.getPix(multiIm, annotationIm[:,:,1])
[meatPix, meatR, meatC] = hf.getPix(multiIm, annotationIm[:,:,2])

# Plot mean and std values for meat and fat pixels for all layers
plt.plot(np.mean(meatPix,0),'b', label='Meat')
plt.plot(np.mean(fatPix,0),'r', label='Fat')
plt.title('Mean')
plt.xlabel('Layer')
plt.ylabel('Value')
plt.legend()
plt.show()
plt.figure()
plt.plot(np.std(meatPix,0),'b', label='Meat')
plt.plot(np.std(fatPix,0),'r', label='Fat')
plt.title('Standard deviation')
plt.xlabel('Layer')
plt.ylabel('Value')
plt.legend()
plt.show()

# Plot distribution of meat and fat pixel values for layer 3, 10 and 16
plt.figure()
plt.hist(meatPix[:,2],bins=19)
plt.hist(meatPix[:,9],bins=19)
plt.hist(meatPix[:,15],bins=19)
plt.title('Distribution of meat')
plt.show()
plt.figure()
plt.hist(fatPix[:,2],bins=19)
plt.hist(fatPix[:,9],bins=19)
plt.hist(fatPix[:,15],bins=19)
plt.title('Distribution of fat')
plt.show()


# Plot normal (assumption) distribution of meat and fat for day 1, layer 1
x = np.linspace(1, 100, 1000)
y_meat = norm.pdf(x, np.mean(meatPix,0)[0], np.std(meatPix,0)[0])  # Probability density function (PDF)
y_fat = norm.pdf(x, np.mean(fatPix,0)[0], np.std(fatPix,0)[0])  # Probability density function (PDF)
plt.plot(x, y_meat, label='Meat')
plt.plot(x, y_fat, label='Fat')
plt.title('Normal Distribution')
plt.xlabel('Value')
plt.ylabel('Probability Density')
plt.legend()
plt.show()