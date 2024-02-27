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

fat_mat = multiIm[annotationIm[:,:,1] == 1]
meat_mat = multiIm[annotationIm[:,:,2] == 1]

print(fat_mat.shape)

def calculateCovarianceMatrix(matrix):
    m = matrix.shape[0]
    sigma = np.cov(matrix.T)
    return sigma, m

def calculatePoolSigma (matrix1, matrix2):
    sigma1, m1 = calculateCovarianceMatrix(matrix1)
    sigma2, m2 = calculateCovarianceMatrix(matrix2)
    pool_sigma = 1/(m1-1+m2-1) * ((m1-1)*sigma1 + (m2-1)*sigma2)
    return pool_sigma

def S(x,mu,sigma,p):
    return x.T*np.linalg.inv(sigma)*mu-1/2*mu.T*np.linalg.inv(sigma)*mu+np.log(p)