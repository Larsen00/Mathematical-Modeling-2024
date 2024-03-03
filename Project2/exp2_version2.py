import os
import sys
import platform
import helpFunctions as hf 
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.pyplot import imread 
from scipy.stats import norm

## Example of loading a multi spectral image
dirIn = rf'{os.getcwd()}/Project2/data/'  

# Load multi spectral image and annotations from day 1
multiIm, annotationIm = hf.loadMulti('multispectral_day01.mat' , 'annotation_day01.png', dirIn)

# Matrix of fat and meat repsectively
fat_mat = multiIm[annotationIm[:,:,1] == 1]
meat_mat = multiIm[annotationIm[:,:,2] == 1]

# Calculates the Covariance matrix for given matrix
def calculateCovarianceMatrix(matrix):
    m = matrix.shape[0]
    sigma = np.cov(matrix.T)
    return sigma, m


def calculatePooledCorvarianceMatrix(matrix1, matrix2):
    sigma1, m1 = calculateCovarianceMatrix(matrix1)
    sigma2, m2 = calculateCovarianceMatrix(matrix2)
    pool_sigma = 1/(m1-1+m2-1) * ((m1-1)*sigma1 + (m2-1)*sigma2)
    return pool_sigma

# multivariate linear discriminant function
def S(x,mu,sigma,p):
    return x.T@np.linalg.inv(sigma)@mu-1/2*mu.T@np.linalg.inv(sigma)@mu+np.log(p)

# Extract fat and meat pixel values
[fatPix, fatR, fatC] = hf.getPix(multiIm, annotationIm[:,:,1])
[meatPix, meatR, meatC] = hf.getPix(multiIm, annotationIm[:,:,2])

def calculateErrorRate(matrix1, matrix2, p1, p2):
    '''
    Calculate the error rate (disagreement between the model and the annotations) for the training set.
    Input:  matrix1 is the fat pixels, 
            matrix2 is the meat pixels, 
            p1 and p2 are the prior probilities for fat and meat pixels.
    Return: total error rate, error rate for matrix1 and error rate for matrix2.
    '''
    error1 = 0
    error2 = 0
    total = matrix1.shape[0] + matrix2.shape[0]
    
    # finds the pooled covariance matrix
    sigma = calculatePooledCorvarianceMatrix(matrix1,matrix2)

    # finds the mean of the two matrixes for each layer
    mu1 = np.mean(matrix1, axis=0).T
    mu2 = np.mean(matrix2, axis=0).T

    # calculates the error for the first matrix
    for i in range(0, matrix1.shape[0]):
        x = matrix1[i,:].T
        prob1 = S(x, mu1, sigma, p1)
        prob2 = S(x, mu2, sigma, p2)
        if prob2 > prob1:
            error1 += 1

    # calculates the error for the second matrix
    for i in range(0, matrix2.shape[0]):
        x = matrix2[i,:].T
        prob1 = S(x, mu1, sigma, p1)
        prob2 = S(x, mu2, sigma, p2)
        if prob2 < prob1:
            error2 += 1
            
    # returns the total error rate, error rate for matrix1 and error rate for matrix2
    return (error1+error2)/total, error1/matrix1.shape[0], error2/matrix2.shape[0]


errorrate, error_fat, error_meat = calculateErrorRate(fatPix, meatPix, 0.30, 0.70)
print(f'total error rate: {errorrate}, \nerror rate for fat: {error_fat}, \nerror rate for meat: {error_meat}')

# find salami
plt.imshow(np.sum(annotationIm, axis=2))
plt.show()

mu1 = np.mean(fatPix, axis=0).T
mu2 = np.mean(meatPix, axis=0).T
p1 = 0.3
p2 = 1-p1
sigma = calculatePooledCorvarianceMatrix(fatPix,meatPix)
binIm = np.zeros((multiIm.shape[0],multiIm.shape[1]))
for i in range(multiIm.shape[0]):
    for j in range(multiIm.shape[1]):
        x = multiIm[i,j,:].T
        prob1 = S(x, mu1, sigma, p1)
        prob2 = S(x, mu2, sigma, p2)
        if prob1 > prob2:
            binIm[i,j] = 1
plt.imshow(binIm)
plt.show()