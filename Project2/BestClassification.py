from exp2_version2 import calculateErrorRate
from ex2 import ShowClassifyIm
import helpFunctions as hf 
import os
import matplotlib
from matplotlib import pyplot as plt
import numpy as np
import seaborn as sns


dirIn = rf'{os.getcwd()}/Project2/data/'  
#Containers for classified images and the original png's
container_classified_ims = []
original_ims = []

days = ["01", "06", "13", "20", "28"]

#Only want to train on day 28 as that one is the best
for training_image in days[4:5]:

    # loads the training image with the annotation image
    multiIm_ti, annotationIm_ti = hf.loadMulti(f'multispectral_day{training_image}.mat' , f'annotation_day{training_image}.png', dirIn)

    for test_image in days:

        # Skip if training and test image is the same
        if training_image == test_image:
            continue
        

        # Load multi spectral image and annotations from day 1
        multiIm, annotationIm = hf.loadMulti(f'multispectral_day{test_image}.mat' , f'annotation_day{test_image}.png', dirIn)

        # Load png image of the spegepølse
        original_ims.append(matplotlib.image.imread(f'{dirIn}color_day{test_image}.png'))
        # Matrix of fat and meat repsectively

        fat_mat = multiIm[annotationIm[:,:,1] == 1]
        meat_mat = multiIm[annotationIm[:,:,2] == 1]

        # Use ShowClassifyIm from ex2 to classify the sausage from previous days
        imageclassified = ShowClassifyIm(
            multiIm,
            annotationIm,        
            multiIm_ti[annotationIm_ti[:,:,1] == 1],
            multiIm_ti[annotationIm_ti[:,:,2] == 1],
            0.3, 0.7,
        )
        container_classified_ims.append(imageclassified)
titles = ['Day 1', 'Day 6', 'Day 13', 'Day 20']  # Titles for each subplot

fig, axs = plt.subplots(2, 4, figsize=(20, 10))  # Create a 2x4 grid of subplots
colors = ['purple', 'yellow', 'blue']
labels = ['Background', 'Meat', 'Fat'] 

# Create custom patches as handles for the legend
patches = [matplotlib.patches.Patch(color=colors[i], label=labels[i]) for i in range(len(colors))]

# Plot the original png's in the first row
for i in range(4):
    axs[0, i].imshow(original_ims[i])
    axs[0, i].set_title(f'Original Image {titles[i]}')
    axs[0, i].axis('off')  # Hide axes for cleaner look

# Plot the classified images in the second row
for i in range(4):
    axs[1, i].imshow(container_classified_ims[i], cmap='viridis')
    axs[1, i].set_title(f'Classified Image {titles[i]}')  # Set a title for each subplot
    axs[1, i].axis('off')  # Hide axes for cleaner look

fig.legend(handles=patches, loc='lower center', ncol=3)

plt.suptitle('Training Day 28')
plt.tight_layout()  # Adjust the spacing between plots
#plt.colorbar(im, ax=axs, orientation='horizontal', fraction=0.02, pad=0.04)  # Add a colorbar
path = './Project2/plots/training_day_28.png'
fig.savefig(path, dpi=300, bbox_inches='tight')