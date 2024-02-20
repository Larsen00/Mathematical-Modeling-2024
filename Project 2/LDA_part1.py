import sympy as s
from sympy.plotting import plot
import matplotlib.pyplot as plt
import numpy as np

#%%  
# 2. Plot in a coordinate system the probability density function (PDF) for the two values of μ.
# define x range
x_lower, x_upper = 140, 250
x = np.arange(x_lower, x_upper,0.1)
# define mu values
mu1 = 175.5
mu2 = 162.9
# define PDF
f1 = lambda x: np.exp(-(x-mu1)**2/(2*(6.7**2)))/(6.7*np.sqrt(2*np.pi))
f2 = lambda x: np.exp(-(x-mu2)**2/(2*(6.7**2)))/(6.7*np.sqrt(2*np.pi))
# plot PDF
plt.plot(x, f1(x))
plt.plot(x, f2(x))
plt.xlim(x_lower, x_upper)
plt.show()