import glob
from skimage import io, color
import numpy as np
import time
import os
import scipy
import matplotlib.pyplot as plt


# impotere 
dir = os.getcwd()
paths = glob.glob(dir + '/toyProblem_F22/*.png')
images = [color.rgb2gray(io.imread(path)) for path in paths]


# for img in images:
#     io.imshow(img)
#     io.show()
#     time.sleep(1)

# Problem 2.1
im = images[0]
dx = [im[:, 1:] - im[:, 0:-1] for im in images]
dy = [im[1:, :] - im[0:-1, :] for im in images]
dt = [im2 - im1 for im1, im2 in zip(images[0:-1], images[1:])]
# display for selected frames
# i want to show the 10th frame
# i want to show 4 images in one plot
plt.figure(figsize=(10, 8))  # Adjust the figure size as needed
plt.subplot(2, 2, 1)
plt.imshow(images[10],cmap='gray')
plt.title('Frame 10 - Original Image')
plt.subplot(2, 2, 2)
plt.imshow(dy[10],cmap='gray')
plt.title('dy')
plt.subplot(2, 2, 3)
plt.imshow(dx[10],cmap='gray')
plt.title('dx')
plt.subplot(2, 2, 4)
plt.imshow(dt[10],cmap='gray')
plt.title('dt')
plt.show()
print("Works")

# Problem 2.2
# Create a Prewitt kernel for the horizontal gradient
kernel = np.array([[-1, 0, 1],
                   [-1, 0, 1],
                   [-1, 0, 1]])

# Display the kernel as an image
plt.imshow(kernel, cmap='gray')
plt.title('Prewitt Kernel for Horizontal Gradient')
plt.colorbar()
plt.show()


sobel_horizontal = np.array([[-1, 0, 1],
                              [-2, 0, 2],
                              [-1, 0, 1]])

sobel_vertical = np.array([[-1, -2, -1],
                            [0, 0, 0],
                            [1, 2, 1]])

# Create a 1x2 grid of subplots to display both kernels
plt.figure(figsize=(12, 5))

# Plot the Sobel kernel for horizontal gradient
plt.subplot(1, 2, 1)
plt.imshow(sobel_horizontal, cmap='gray')
plt.title('Sobel Horizontal Kernel')
plt.colorbar()

# Plot the Sobel kernel for vertical gradient
plt.subplot(1, 2, 2)
plt.imshow(sobel_vertical, cmap='gray')
plt.title('Sobel Vertical Kernel')
plt.colorbar()

plt.tight_layout()
plt.show()
## take one image
im = images[0]
## use prewitt filter
prewitt = scipy.ndimage.prewitt_h(im)
io.imshow(prewitt, cmap='gray')
io.show()
## use sobel filter
sobel = scipy.ndimage.sobel(im)
io.imshow(sobel, cmap='gray')
io.show()






