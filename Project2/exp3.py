from exp2_version2 import calculateErrorRate
import helpFunctions as hf 
import os
from matplotlib import pyplot as plt
import numpy as np
import seaborn as sns


def threshold_same_variance(m1, m2):
    return (m1 + m2) / 2

def finds_best_band_using_threshold(fatPix, meatPix):
    t = []
    # calculating the mean and standard deviation of meat and fat pixels
    mean_meat = np.mean(meatPix, 0)
    mean_fat = np.mean(fatPix, 0)

    for i in range(0, 19):
        i_point = threshold_same_variance(mean_fat[i], mean_meat[i])
        t.append(i_point)

    # 2. Calculate the error rate for each spectral band.
    error_rate_meat = []
    error_rate_fat = []
    for band in range(0, 19):
        if t[band] is not None:
            error_rate_meat.append(sum(1 for i in meatPix[:, band] if i > t[band]))
            error_rate_fat.append(sum(1 for i in fatPix[:, band] if i < t[band]))

    # 3. Identify the spectral band, which has the best discriminative properties for meat and fat.
    mean_error_rate = []
    for i in range(0, 19):
        if t[i] is not None:
            mean_error_rate.append((error_rate_meat[i] + error_rate_fat[i]) / 2)

    best_band = mean_error_rate.index(min(mean_error_rate))
    return best_band, t[best_band], mean_fat[best_band] > mean_meat[best_band]



def cal_thres_errorrate(fatPix, meatPix, band, threshold, mean_meat_less_mean_fat):

    # recall meat is less than threshold, fat is greater than threshold
    if mean_meat_less_mean_fat:
        thres_errors_fat = np.sum(fatPix[:, band] < threshold)
        thres_errors_meat = np.sum(meatPix[:, band] > threshold)
    else :
        thres_errors_fat = np.sum(fatPix[:, band] > threshold)
        thres_errors_meat = np.sum(meatPix[:, band] < threshold)    
    return (thres_errors_fat + thres_errors_meat) / (fatPix.shape[0] + meatPix.shape[0])


## Example of loading a multi spectral image
dirIn = rf'{os.getcwd()}/Project2/data/'  

days = ["01", "06", "13", "20", "28"]
average_error_rates = []
error_rates = []
average_error_rates_thres = []
error_rates_thres = []


best_bands = []
for training_image in days:

    # loads the training image
    multiIm_ti, annotationIm_ti = hf.loadMulti(f'multispectral_day{training_image}.mat' , f'annotation_day{training_image}.png', dirIn)
    [fatPix_ti, _, _] = hf.getPix(multiIm_ti, annotationIm_ti[:,:,1])
    [meatPix_ti, _, _] = hf.getPix(multiIm_ti, annotationIm_ti[:,:,2])

    band, threshold, mean_meat_less_mean_fat = finds_best_band_using_threshold(fatPix_ti, meatPix_ti)
    best_bands.append(band)


    error_sum = 0
    errors = []
    error_sum_thres = 0
    errors_thres = []

    for test_image in days:

        # Skip if training and test image is the same
        if training_image == test_image:
            errors.append(np.nan)
            errors_thres.append(np.nan)
            continue
        

        # Load multi spectral image test image
        multiIm, annotationIm = hf.loadMulti(f'multispectral_day{test_image}.mat' , f'annotation_day{test_image}.png', dirIn)      
        
        # Matrix of fat and meat repsectively
        [fatPix, _, _] = hf.getPix(multiIm, annotationIm[:,:,1])
        [meatPix, _, _] = hf.getPix(multiIm, annotationIm[:,:,2])

        # calculate the error rate
        error, _, _ = calculateErrorRate(
            fatPix, meatPix,
            0.3, 0.7,
            fatPix_ti, meatPix_ti
        )
        
        # sum the error rates
        error_sum += error

        # store the error rate
        errors.append(error)


        #### USing a threshold
        
        error_rate_thres = cal_thres_errorrate(fatPix, meatPix, band, threshold, mean_meat_less_mean_fat)
        error_sum_thres += error_rate_thres 
        errors_thres.append(error_rate_thres)
        

    # calculate the average error rate
    average_error_rates.append(error_sum/4)
    average_error_rates_thres.append(error_sum_thres/4)

    # store the error rates
    error_rates.append(errors)
    error_rates_thres.append(errors_thres)


###### LDA ########
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
ax.set_title('Error rates using LDA')
plt.show()



###### Threshold ########
# find the best day for training, according to the average error rate
best_day_thres = average_error_rates_thres.index(min(average_error_rates_thres))
print(f'Best day for training using threshold: {days[best_day_thres]} with an average error rate of {average_error_rates_thres[best_day_thres]}')
print(best_bands)

# Plotting the error rates
corralation_matrix_thres = np.matrix(error_rates_thres)
corralation_matrix_thres = np.c_[corralation_matrix_thres, average_error_rates_thres]
x_name = days + ["average_error"]
ax = sns.heatmap(corralation_matrix_thres, annot = True, cmap ='plasma', 
            linecolor ='black', linewidths = 1, xticklabels=x_name, yticklabels=days)

ax.set_xlabel('Validation day')
ax.set_ylabel('Training day')
ax.set_title('Error rates using threshold')
plt.show()