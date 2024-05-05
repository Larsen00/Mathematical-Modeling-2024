import numpy as np
import scipy.ndimage
import matplotlib.pyplot as plt
import os
from load_images import load_images
import scipy.ndimage as ndimage

def make_dir(path:str):
    """
    Create a directory at the specified path if it doesn't already exist.

    Parameters:
    path (str): The path of the directory to be created.

    Returns:
    None
    """
    if os.path.isdir(path) == False:
        os.mkdir(path)
    return

def Lucas_Kanade_method(V:np.ndarray, sigma:float=1, n:int=3, stride:int=1, objects=[]) -> np.ndarray:
    """
    Apply Gaussian filter and use Lucas-Kanade method to calculate optical flow.
    ---
    Args:
        V: 3D array of images (y, x, t)
        sigma: Sigma value for Gaussian filter
        n: Neighbourhood size
        stride: Stride size
        objects: List of objects to calculate optical flow
    Return:
        Results of optical flow in the form of a 4D array (2, V.shape[0], V.shape[1], V.shape[2])
    """
    # Apply gaussian filter on each axis
    print(f"Creating Gaussian filter with sigma = {sigma}...")
    Vy_gaussian = scipy.ndimage.gaussian_filter(V, sigma, order=1, axes=0)
    Vx_gaussian = scipy.ndimage.gaussian_filter(V, sigma, order=1, axes=1)
    Vt_gaussian = scipy.ndimage.gaussian_filter(V, sigma, order=1, axes=2)
    print("Gaussian filter done!")
    print("Calculating optical flow...")
    print(f"\tNeighbourhood size: {n}")
    if objects == []:
        objects = range(V.shape[2])
    total_volume = V.shape[0]*V.shape[1]*len(objects)
    print("\tPreallocating memory...")
    dps_images = np.zeros((2, V.shape[0], V.shape[1], len(objects)))
    print("\tPreallocating memory done!")
    # Perform Lucas-Kanade
    for i, t in enumerate(objects):
        dps_image = np.zeros((2, V.shape[0], V.shape[1]))
        for y in range(0,V.shape[0],stride):
            for x in range(0,V.shape[1],stride):
                x_lower, x_upper = np.maximum(0, x-n), np.minimum(V.shape[1], x+n+1)
                y_lower, y_upper = np.maximum(0, y-n), np.minimum(V.shape[0], y+n+1)
                dp = np.linalg.lstsq(np.column_stack((Vx_gaussian[y_lower:y_upper, x_lower:x_upper, t].ravel(), Vy_gaussian[y_lower:y_upper, x_lower:x_upper, t].ravel())),-Vt_gaussian[y_lower:y_upper, x_lower:x_upper, t].ravel(),rcond=None)[0]
                dps_image[:,y,x] = dp
                # print(f"Progress: {(i*V.shape[0]*V.shape[1] + y*V.shape[1] + x)/total_volume*100:.2f}%", end="\r")
        dps_images[:,:,:,i] = dps_image
    print("Lucas Kanade method done!")
    return dps_images
    
def plot_with_noise_filtering(dps_images:np.ndarray, V:np.ndarray, timesDay, times, mask, threshold:float=1.0, stride:int=1, show:bool=False) -> np.ndarray:
    """
    Filter out noises in the optical flow.
    ---
    Args:
        dps_images: Results of optical flow in the form of a 4D array (2, V.shape[0], V.shape[1], V.shape[2])
        V: 3D array of images after gaussian filtering (y, x, t)
        timesDay: List of dates
        times: List of times
        threshold: Threshold value to filter out noises
        stride: Stride size
        show: Whether to show the plot
    Return:
        Plot filtered optical flow
    """
    if show == False:
        make_dir(f"{path}/_static")
    print("Filtering out noises and plot...")
    # plot with noises filtered out 
    dps_images_filtered = dps_images.copy()
    # for t in range(V.shape[2]):
    # Load binary mask outlining Denmark
    mask = np.load(f'{path}/mask.npy')
    # Register the number of dimensions
    n_y_dims, n_x_dims, n_t_dims = dps_images_filtered.shape[1:4]
    print(f"Number of dimensions: {n_y_dims}x{n_x_dims}x{n_t_dims}")
    # Apply threshold
    for t in range(n_t_dims):
        plt.imshow(V[:,:,t]*mask, cmap="gray")
        for y in range(0,n_y_dims,stride):
            for x in range(0,n_x_dims,stride):
                ## filter out arrows with small magnitude (<= 5)
                tmp = np.linalg.norm(dps_images_filtered[:,y,x,t])
                if tmp > threshold and mask[y,x] == True:
                    if (dps_images_filtered[0,y,x,t] > 5 or dps_images_filtered[1,y,x,t] > 5):
                        print(dps_images_filtered[0,y,x,t], dps_images_filtered[1,y,x,t])
                    plt.quiver(x, y, dps_images_filtered[0,y,x,t], dps_images_filtered[1,y,x,t], color="red", width=.002)
        plt.tight_layout()
        plt.title(f"Optical Flow Filtered 202403{timesDay[t]}_{times[t]}")
        if show:
            plt.pause(0.5)
        else:
            plt.savefig(f"{path}/_static/flow_filtered_202403{timesDay[t]}_{times[t]}.png")
        plt.clf()
        print(f"Plotting image {t+1}/{n_t_dims}.", end="\r")
    print("All Done! Program terminates.")

def interpolate_flow(dps_images:np.ndarray, V:np.ndarray, timesDay, times, mask, batch_size:int=0, n:int=1, objects=[], show:bool=True):
    """
    Interpolate the flow between two images.
    ---
    Args:
        dps_images: Results of optical flow in the form of a 4D array (2, V.shape[0], V.shape[1], V.shape[2])
        V: 3D array of images after gaussian filtering (y, x, t)
        timesDay: List of dates
        times: List of times
        batch_size: Size of the batch of pixels around each pixel which will be moved with each individual pixel
        mask: Binary mask outlining Denmark
        n: Number of interpolated images(include last image)
        objects: List of objects to interpolate
        show: Whether to show the plot
    Return:
        return interpolated images from based image to the last interpolated image, in one 3D array.
    """

    if objects == []:
        objects = range(V.shape[2])

    # interpolate flow
    V_interpolation = np.zeros((V.shape[0], V.shape[1], n*len(objects)))
    timestampss = []
    l = 0   # l is a variable to increment every time an image is added to V_interpolation
    #Interpolation starting here:
    for i, t in enumerate(objects):
        interpolate_images = [] #Container for the current interpolated image 
        timestamps = []
        original_image = V[:,:,t] #Iterate across all original images
        V_interpolation[:,:,l] = original_image #Make sure that the original images is also part of the final interpolated dataset. 
        l += 1
        for j in range(n-1):
            interpolate_image = np.zeros_like(original_image).copy()
            interpolate_image[:] = np.nan
            for y in range(dps_images.shape[1]):
                for x in range(dps_images.shape[2]):
                    if (dps_images[0,y,x,i] == 0 or dps_images[1,y,x,i] == 0):
                        continue
                    else:
                        # move pixel
                        dx = np.rint(dps_images[0,y,x,i]*(j+1)/(n)).astype(int)*2
                        dy = np.rint(dps_images[1,y,x,i]*(j+1)/(n)).astype(int)*2
                        # move pixels as a batch to same direction
                        if not (dx == 0 and dy == 0):
                            x_lower, x_upper = np.maximum(0, x-batch_size), np.minimum(V.shape[1], x+batch_size+1)
                            y_lower, y_upper = np.maximum(0, y-batch_size), np.minimum(V.shape[0], y+batch_size+1)
                            for y_batch in range(y_lower, y_upper):
                                for x_batch in range(x_lower, x_upper):
                                    #Move pixel according to displacement
                                    move_pixel(interpolate_image, original_image, mask, (x_batch, y_batch), (x_batch+dx, y_batch+dy))
            # Fill nan values from the earth image
            fill_image(interpolate_image, mask, V[:,:,t])
            V_interpolation[:,:,l] = interpolate_image
            l += 1
            interpolate_images.append(interpolate_image)

        timestamp = times[t]
        timestamps.append(timestamp)       
        for k, img in enumerate(interpolate_images):
            hour = int(times[t][:2])
            minutes = (k+1)*(15//n) + int(times[t][2:4])
            if minutes >= 60:
                minutes -= 60
                hour += 1
            timestamp = f"{str(hour).zfill(2)}{str(minutes).zfill(2)}{times[t][4:]}"
            timestamps.append(timestamp)
            if show:

                plt.imshow(img*mask, cmap="viridis")
                plt.title(f"202403{timesDay[t]}_" + timestamp)
                plt.pause(0.5)
                plt.clf()
        # interpolate_imagess.append(interpolate_images)
        timestampss.append(timestamps)
    pass
    return V_interpolation, (np.array(timestampss).ravel()).tolist()

def move_pixel(img, img_origin, mask, source, target) -> None:
    """
    Move pixel from source to target
    ---
    Args:
        img: 2D array of image
        img_origin: 2D array of original image
        source: (x, y) coordinate of source pixel
        target: (x, y) coordinate of target pixel
    Return:
        None
    """
    if not (target[0] >= img.shape[1] or target[1] >= img.shape[0] or target[0] < 0 or target[1] < 0):
        if mask[target[1], target[0]] == 1.0:
            if np.isnan(img[target[1], target[0]]):
                #Move pixel to target
                img[target[1], target[0]] = img_origin[source[1], source[0]]
                # print(f"Moving pixel from {source} to {target}, pixel value {img_origin[source[1], source[0]]}")
            # else:
            #     img[target[1], target[0]] = np.minimum(img[target[1], target[0]], img_origin[source[1], source[0]]) - np.abs(img[target[1], target[0]] - img_origin[source[1], source[0]])
    return

def fill_image_with_avg(interpolate_image, mask, kernel_size=10, use_global_avg=False):
    """
    Fills NaN values in the given image using average values from a specified neighborhood defined by kernel_size.
    This operation is performed only in the areas specified by the mask.

    Parameters:
        interpolate_image (np.array): The image array containing NaN values to fill.
        mask (np.array): A boolean array where True indicates areas that should be considered for NaN filling.
        kernel_size (int): The size of the neighborhood to consider for the local average calculation.
        use_global_avg (bool): If True, use the global average of the image to fill any remaining NaNs after local avg filling.

    Returns:
        None; the interpolate_image is modified in place.
    """
    
    # Handle NaNs: Create a masked array that ignores NaNs for averaging
    masked_image = np.ma.array(interpolate_image, mask=np.isnan(interpolate_image))
    
    # Apply uniform filter to find the neighborhood average on the valid (non-NaN) parts of the image
    local_avg = ndimage.uniform_filter(masked_image.filled(0), size=kernel_size, mode='constant', cval=0)
    
    # Count of non-NaN entries within each filter area to correct areas with few non-NaNs contributing to the average
    count_non_nan = ndimage.uniform_filter(~masked_image.mask, size=kernel_size, mode='constant', cval=0)
    
    # Calculate corrected local average
    corrected_avg = local_avg / count_non_nan
    
    # Replace NaNs where necessary: Only in masked areas
    fill_positions = np.isnan(interpolate_image) & mask
    interpolate_image[fill_positions] = corrected_avg[fill_positions]

    # If global average is needed and any NaN remains
    if use_global_avg and np.isnan(interpolate_image).any():
        # Calculate the global average from the valid entries in the image
        global_avg = np.nanmean(interpolate_image)
        # Fill any remaining NaNs with the global average
        interpolate_image[np.isnan(interpolate_image)] = global_avg



def fill_image_with_max(interpolate_image, mask, kernel_size=10, use_global_max=False):

    temp_image = np.where(np.isnan(interpolate_image), -np.inf, interpolate_image)
    
    # Apply maximum filter to find the neighborhood maximum
    neighborhood_max = ndimage.maximum_filter(temp_image, size=kernel_size, mode='constant', cval=np.nan)

    # Only fill NaNs where the mask is True
    fill_positions = np.isnan(interpolate_image) & mask
    
    # Replace NaNs with the maximum value from their neighborhood
    interpolate_image[fill_positions] = neighborhood_max[fill_positions]
    
    # Check if any NaNs remain and use_global_max is set to True
    if use_global_max and np.isnan(interpolate_image).any():
        # Calculate the global maximum from the valid entries in the image
        global_max = np.nanmax(interpolate_image)
        # Fill any remaining NaNs with the global maximum
        interpolate_image[np.isnan(interpolate_image)] = global_max
    

def predict(i, dps_images, original_image, interval_n, interval_size, batch_size, mask, fill_method):
    next_image = np.zeros_like(original_image).copy()
    next_image[:] = np.nan
    same = 0
    for y in range(dps_images.shape[1]):
        for x in range(dps_images.shape[2]):
            if (dps_images[0,y,x,i] == 0 or dps_images[1,y,x,i] == 0):
                continue
            else:
                # move pixel
                dx = np.rint(dps_images[0,y,x,i]*interval_n/(interval_size)).astype(int)
                dy = np.rint(dps_images[1,y,x,i]*interval_n/(interval_size)).astype(int)
                
                # move pixels as a batch to same direction
                x_lower, x_upper = np.maximum(0, x-batch_size), np.minimum(original_image.shape[1], x+batch_size+1)
                y_lower, y_upper = np.maximum(0, y-batch_size), np.minimum(original_image.shape[0], y+batch_size+1)
                
                if dx == 0 and dy == 0:
                    same += 1
                    
                for y_batch in range(y_lower, y_upper):
                    for x_batch in range(x_lower, x_upper):
                        move_pixel(next_image, original_image, mask, (x_batch, y_batch), (x_batch+dx, y_batch+dy))
    
    # # Create a masked array where True marks NaNs in the original image
    # nan_mask = np.isnan(next_image)

    # # Use logical AND to find where both the mask is True and the image is NaN
    # masked_nans = nan_mask & mask

    # # Count the True values in the masked_nans array
    # count = np.count_nonzero(masked_nans) + 1
    # tot = np.count_nonzero(mask)
    # print(f'The number of pixels is same {count} out of {tot} which is {count/tot}% ')

   
    # fill nans
    if fill_method == 1:
        fill_image_with_max(next_image, mask, 20, True)
    elif fill_method == 2:
        fill_image_with_avg(next_image, mask, 20, True)
    elif fill_method == 3:
        fill_image(next_image, mask, original_image)
    else :
        raise Exception("fill method dont exist")
    
    
    return next_image
def extrapolate_flow(dps_images:np.ndarray, V:np.ndarray, timesDay, times, mask, minutes_after:int=15, batch_size:int=0, objects=[], show:bool=False, fill_method:int=1) -> list:
    """
    extrapolate the flow between two images.
    ---
    Args:
        dps_images: Results of optical flow in the form of a 4D array (2, V.shape[0], V.shape[1], V.shape[2])
        V: 3D array of images after gaussian filtering (y, x, t)
        timesDay: List of dates
        times: List of times
        mask: Binary mask outlining Denmark
        minutes_after: Minutes after the base image
        batch_size: Size of the batch of pixels around each pixel which will be moved with each individual pixel
        objects: List of objects to extrapolate
        show: Whether to show the plot
    Return:
        List of extrapolated images with timestamps
    """
    # extrapolate flow
    extrapolate_images = []
    for i, t in enumerate(objects):
        original_image = V[:,:,t]
        
        # Fill nan values from the earth image
        # fill_image(extrapolate_image, mask, V[:,:,t])
        extrapolate_image = predict(i, dps_images, original_image, minutes_after, 15, 0, mask, fill_method)
        extrapolate_images.append(extrapolate_image)
        
    timestamps = []
    for i, t in enumerate(objects):
        hour = int(times[t][:2])
        minutes = minutes_after + int(times[t][2:4])
        if minutes >= 60:
            minutes -= 60
            hour += 1
        timestamps.append(f"{str(hour).zfill(2)}{str(minutes).zfill(2)}{times[t][4:]}" )
        if show:
            fig, axs = plt.subplots(1, 3, figsize=(16, 9))
            axs[0].imshow(extrapolate_images[i]*mask, cmap="viridis")
            axs[0].set_title(f"Extrapolated Image 202403{timesDay[t]}_{times[t][:2]}{str(minutes_after + int(times[t][2:4])).zfill(2)}{times[t][4:]}")
            axs[1].imshow((V[:,:,t + 1])*mask, cmap="viridis")
            axs[1].set_title(f"Original Image 202403{timesDay[t+1]}_{times[t+1]}")
            axs[2].imshow((extrapolate_images[i] - V[:,:,t + 1])*mask, cmap="viridis")
            axs[2].set_title(f"Difference, MSE: {MSE(extrapolate_images[i], V[:,:,t + 1], mask):.2f}, Baseline MSE: {MSE(V[:,:,t], V[:,:,t + 1], mask):.2f}")
            plt.tight_layout()
            plt.show()
    return np.dstack(extrapolate_images), timestamps

import numpy as np

def MSE(y_true, y_pred, mask):
    """
    Calculates the mean squared error (MSE) between the true values and predicted values.

    Parameters:
    - y_true (ndarray): Array of true values.
    - y_pred (ndarray): Array of predicted values.
    - mask (ndarray): Boolean mask to only consider pixels representing Danish soil.

    Returns:
    - mse (float): Mean squared error between the true and predicted values.
    """

    # Calculate the squared difference between true and predicted values
    squared_diff = np.square(y_true[mask] - y_pred[mask])

    # Sum the squared differences and divide by the number of masked values
    mse = squared_diff.sum() / mask.sum()

    return mse

import matplotlib.pyplot as plt

def plt_imshow(img):
    """
    Display an image using matplotlib's imshow function.

    Parameters:
    img (numpy.ndarray): The image to be displayed.

    Returns:
    None
    """
    # Display the image using the "viridis" colormap
    plt.imshow(img, cmap="viridis")
    plt.show()
    return

def fill_image(img, mask, background_image):
    """
    Fill the NaN values in the image with corresponding values from the background image.

    Parameters:
    - img (ndarray): The image to be filled.
    - mask (ndarray): Boolean mask to only consider pixels representing Danish soil.
    - background_image (ndarray): The background image.

    Returns:
    None
    """
    for y in range(img.shape[0]):
        for x in range(img.shape[1]):
            if np.isnan(img[y,x]) and mask[y,x] == 1.0:
                img[y,x] = background_image[y,x]
            elif mask[y,x] == 0.0:
                img[y,x] = 0.0


def cal_MSE_forall():
    path = 'Processedfull'
    files_in_directory = os.listdir(path)
    dates = [file.replace('.xlsx', '') for file in files_in_directory if file.endswith('.xlsx')]
    mse_all_pred_max = []
    mse_all_pred_avg = []
    mse_all_pred_sim = []
    mse_all_last = []
    mse_all_toLast = []
    
    for target in dates:
        V, timesDay, times, mask = load_images(target, path)
        objects = range(V.shape[2] - 1)
        dps_images = Lucas_Kanade_method(V, objects=objects)
    
        # E_max, _ = extrapolate_flow(dps_images, V, timesDay, times, mask, minutes_after=15, objects=objects, show=False, fill_method=1)
        # E_avg, _ = extrapolate_flow(dps_images, V, timesDay, times, mask, minutes_after=15, objects=objects, show=False, fill_method=2)
        E_sim, _ = extrapolate_flow(dps_images, V, timesDay, times, mask, minutes_after=15, objects=objects, show=False, fill_method=3)
    
        # mse_pred_max = []
        # mse_pred_avg = []
        mse_pred_sim = []
        mse_last = []
        mse_toLast = []
    
        for i in range(E_sim.shape[2]):
            # mse_pred_max.append(MSE(V[:, :, i+1], E_max[:, :, i], mask))
            # mse_pred_avg.append(MSE(V[:, :, i+1], E_avg[:, :, i], mask))
            mse_pred_sim.append(MSE(V[:, :, i+1], E_sim[:, :, i], mask))
            mse_last.append(MSE(V[:, :, i+1], V[:, :, i], mask))
            mse_toLast.append(MSE(V[:, :, i], E_sim[:, :, i], mask))
    
        # mse_all_pred_max.append(np.mean(mse_pred_max))
        # mse_all_pred_avg.append(np.mean(mse_pred_avg))
        mse_all_pred_sim.append(np.mean(mse_pred_sim))
        mse_all_last.append(np.mean(mse_last))
        mse_all_toLast.append(np.mean(mse_toLast))
    
        # print(f"Mean squared error for max fill pred img {target}: {np.mean(mse_pred_max)}")
        # print(f"Mean squared error for avg fill pred img {target}: {np.mean(mse_pred_avg)}")
        print(f"Mean squared error for sim fill pred img {target}: {np.mean(mse_pred_sim)}")
        print(f"Mean squared error for last img {target}: {np.mean(mse_last)}")
    

    # Assume x and MSE data arrays are defined
    x = [day[6:8] for day in dates]  # Extract day from date string assuming 'dates' is defined
    
    plt.figure(figsize=(10, 6))  # Adjust figure size to your preference
    plt.plot(x, mse_all_pred_sim, label='Predicted to Validation Image', marker='o', linestyle='-', markersize=8)
    plt.plot(x, mse_all_last, label='Original to Validation Image', marker='o', linestyle='-', markersize=8)
    plt.plot(x, mse_all_toLast, label='Predicted to Original Image', marker='o', linestyle='-', markersize=8)
    
    # Setting a larger title and customizing legend
    plt.title('Comparison of MSE when Extrapolating Images', fontsize=16)  # Larger title
    plt.xlabel('Day in March', fontsize=14)
    plt.ylabel('Mean of MSE', fontsize=14)
    
    # Adjusting ticks for better readability
    plt.xticks(rotation=45, fontsize=12)
    plt.yticks(fontsize=12)
    
    # Enhancing the legend
    plt.legend(fontsize=12, title='Legend', title_fontsize='13', shadow=True, fancybox=True)
    
    plt.grid(True)
    plt.tight_layout()  # Adjust layout to make room for the rotated x-axis labels
    plt.show()





if __name__ == "__main__":

    

    # path = 'Project4/Processedfull'
    # target_days = ['0317']
    # for target in target_days:
    #     V, timesDay, times, mask = load_images(target, path)
    #     print(timesDay, times)
    #     # Define how many objects subject to calculating optical flow
    #     objects=range(5)
    #     dps_images = Lucas_Kanade_method(V)
    #     # plot_with_noise_filtering(dps_images, V, timesDay, times, mask, show=False)

    #     # interpolate_flow(dps_images, V, timesDay, times, mask, objects=objects, show=True, n=3)
    #     # extrapolate_flow(dps_images, V, timesDay, times, mask, minutes_after=15, objects=objects, show=True)