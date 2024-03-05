from exp2_version2 import calculateErrorRate
import helpFunctions as hf 
import os
from matplotlib import pyplot as plt
import numpy as np
import seaborn as sns

## Example of loading a multi spectral image
dirIn = rf'{os.getcwd()}/Project2/data/'  

days = ["01", "06", "13", "20", "28"]
average_error_rates = []
error_rates = []

for training_image in days:

    # loads the training image
    multiIm_ti, annotationIm_ti = hf.loadMulti(f'multispectral_day{training_image}.mat' , f'annotation_day{training_image}.png', dirIn)

    error_sum = 0
    errors = []
    for test_image in days:

        # Skip if training and test image is the same
        if training_image == test_image:
            errors.append(np.nan)
            continue
        

        # Load multi spectral image and annotations from day 1
        multiIm, annotationIm = hf.loadMulti(f'multispectral_day{test_image}.mat' , f'annotation_day{test_image}.png', dirIn)

        # Matrix of fat and meat repsectively
        fat_mat = multiIm[annotationIm[:,:,1] == 1]
        meat_mat = multiIm[annotationIm[:,:,2] == 1]

        # calculate the error rate
        error, _, _ = calculateErrorRate(
            fat_mat, meat_mat,
            0.3, 0.7,
            multiIm_ti[annotationIm_ti[:,:,1] == 1],
            multiIm_ti[annotationIm_ti[:,:,2] == 1]
        )
        
        # sum the error rates
        error_sum += error

        # store the error rate
        errors.append(error)

    # calculate the average error rate
    average_error_rates.append(error_sum/4)

    # store the error rates
    error_rates.append(errors)

# find the best day for training, according to the average error rate
best_day = average_error_rates.index(min(average_error_rates))
print(f'Best day for training: {days[best_day]} with an average error rate of {average_error_rates[best_day]}')

# Plotting the error rates
corralation_matrix = np.matrix(error_rates)
corralation_matrix = np.c_[corralation_matrix, average_error_rates]
x_name = days + ["average_error"]
ax = sns.heatmap(corralation_matrix, annot = True, cmap ='plasma', 
            linecolor ='black', linewidths = 1, xticklabels=x_name, yticklabels=days)

ax.set_xlabel('Validation day')
ax.set_ylabel('Training day')
ax.set_title('Error rates between different training and validation days')
plt.show()