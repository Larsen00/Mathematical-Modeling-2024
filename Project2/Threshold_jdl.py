#%%
# Analysis of data
import os
import helpFunctions as hf 
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.pyplot import imread 
from scipy.stats import norm
## Example of loading a multi spectral image
dirIn = rf'{os.getcwd()}/Project2/data/'  

#%%
# dispalying sementation of day 01
day01 = matplotlib.image.imread(dirIn + 'Annotation_day01.png')
# plt.imshow(day01)
# plt.show()


# 2. What is the spectral distribution of meat and fat respectively?

# loading multispectral image and annotation of day 01
multiIm, annotationIm = hf.loadMulti('multispectral_day01.mat' , 'Annotation_day01.png', dirIn)

# Here is an example with meat- and fat annotation
[fatPix, fatR, fatC] = hf.getPix(multiIm, annotationIm[:,:,1]);
[meatPix, meatR, meatC] = hf.getPix(multiIm, annotationIm[:,:,2]);


# Here we plot the mean values for pixels with meat and fat respectively
# Det er mean value pr layer
plt.plot(np.mean(meatPix,0),'b')
plt.plot(np.mean(fatPix,0),'r')
plt.legend(['Meat', 'Fat'])
plt.show(block=False)
plt.close()


mean_meat = np.mean(np.mean(meatPix,0))
varieance_meat = np.var(np.mean(meatPix,0))
mean_fat = np.mean(np.mean(fatPix,0))
varieance_fat = np.var(np.mean(fatPix,0))
print(f'Mean of meat: {mean_meat}, Variance of meat: {varieance_meat}')
print(f'Mean of fat: {mean_fat}, Variance of fat: {varieance_fat}')

# 3. Is it a reasonable assumption that each pixel is either meat or fat?
# Answer: No, it is not a reasonable assumption that each pixel is either meat or fat. since there are background pixels as well. But it is a fair assumption that each pixel is either meat or fat or background.


#%% Threshold value for a single spectral band
# 1. Calculate the threshold value t for all spectral bands for day 1.

# loading multispectral image and annotation of day 01
multiIm, annotationIm = hf.loadMulti('multispectral_day01.mat' , 'Annotation_day01.png', dirIn)


# https://stackoverflow.com/questions/22579434/python-finding-the-intersection-point-of-two-gaussian-curves
# calculate the intersection of two gaussians
def intersections(m1, std1, m2, std2):
    a = 1/(2*std1**2) - 1/(2*std2**2)
    b = m2/(std2**2) - m1/(std1**2)
    c = m1**2 /(2*std1**2) - m2**2 / (2*std2**2) - np.log(std2/std1)
    points = np.roots([a,b,c])
    for p in points:
        if p >= m1 and p <= m2 or p <= m1 and p >= m2:
            return p
    return None

t = []

# calculating the mean and standard deviation of meat and fat pixels
mean_meat = np.mean(meatPix, 0)
mean_fat = np.mean(fatPix, 0)
std_meat = np.std(meatPix, 0)
std_fat = np.std(fatPix, 0)

# finds thredsholds for all spectral bands using diffrence varince
# for i in range (0, 19):
#     i_point = intersections(mean_meat[i], std_meat[i], mean_fat[i], std_fat[i])
#     t.append(i_point)

def threshold_same_variance(m1, m2):
    return (m1 + m2) / 2

for i in range(0, 19):
    i_point = threshold_same_variance(mean_fat[i], mean_meat[i])
    t.append(i_point)


## NOTES ##
# meat is less than threshold, fat is greater than threshold
# the 19 layer dont have a threshold value
    
# plotting the gaussian distribution of meat and fat pixels for all spectral bands
for i in range(0, 19, 2):
    print(f'Threshold value for spectral band {i+1} is: {t[i]}')
    x = np.linspace(0,100,100)
    plot1=plt.plot(x,norm.pdf(x,mean_meat[i], std_meat[i]))
    plot2=plt.plot(x,norm.pdf(x,mean_fat[i],std_fat[i]))
    if t[i] is not None:
        plot3 = plt.plot(t[i], norm.pdf(t[i],mean_meat[i],std_meat[i]),'o')
    plt.legend(['Meat', 'Fat', 'Intersection'])
    plt.title(f'Gaussian distribution of meat and fat pixels for spectral band {i+1}')
    plt.show()

#%%
# 2. Calculate the error rate for each spectral band.
error_rate_meat = []
error_rate_fat = []
for band in range(0, 19):
    if t[band] is not None:
        error_rate_meat.append(sum(1 for i in meatPix[:, band] if i > t[band]))
        error_rate_fat.append(sum(1 for i in fatPix[:, band] if i < t[band]))

    

# plotting the error rate for each spectral band
plt.plot(error_rate_meat, 'b')
plt.plot(error_rate_fat, 'r')
plt.legend(['Meat', 'Fat'])
plt.title('Error rate for each spectral band')
plt.show()
    
# 3. Identify the spectral band, which has the best discriminative properties for meat and fat.
mean_error_rate = []
for i in range(0, 19):
    if t[i] is not None:
        mean_error_rate.append((error_rate_meat[i] + error_rate_fat[i]) / 2)

best_band = mean_error_rate.index(min(mean_error_rate)) #14 when using same variance
print(f'The spectral band with the best discriminative properties for meat and fat is: {best_band+1}')
## NOTES
# Det bedste band er layer 1

# 4. Classify the entire image of the salami for day 1, and visualise it.

# Load RGB image
imRGB = imread(dirIn + 'color_day01.png')


# Create a new array where values greater than 't' are assigned 1, and others are assigned 2
clasified_image = np.where(multiIm[:,:,best_band] < t[best_band], 0, 1)
plt.imshow(clasified_image)
plt.title('Classified image of the salami for day 1 using layer {best_band+ 1}')
plt.show()

#%%
