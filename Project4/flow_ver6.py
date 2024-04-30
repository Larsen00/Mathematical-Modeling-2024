# earth image removed
import numpy as np
import scipy.ndimage
import matplotlib.pyplot as plt
import scipy.ndimage as ndimage
import os
from Project4.load_images import load_images

def make_dir(path:str):
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
    # Gaussian filter
    # sigma = int(input("Enter the sigma value for the Gaussian filter: "))
    print(f"Creating Gaussian filter with sigma = {sigma}...")
    # gaussian_V = scipy.ndimage.gaussian_filter(V, sigma, order=1)
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
    # plot with filtering out noises
    dps_images_filtered = dps_images.copy()
    # for t in range(V.shape[2]):
    # Load binary mask outlining Denmark
    mask = np.load(f'{path}/mask.npy')
    # Register the number of dimensions
    n_y_dims, n_x_dims, n_t_dims = dps_images_filtered.shape[1:4]
    print(f"Number of dimensions: {n_y_dims}x{n_x_dims}x{n_t_dims}")
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


def predict(i, dps_images, original_image, interval_n, interval_size, batch_size, mask, fill_method):
    next_image = np.zeros_like(original_image).copy()
    next_image[:] = np.nan
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
                for y_batch in range(y_lower, y_upper):
                    for x_batch in range(x_lower, x_upper):
                        move_pixel(next_image, original_image, mask, (x_batch, y_batch), (x_batch+dx, y_batch+dy))
    
   
    # fill nans
    if fill_method == 1:
        fill_image_with_max(next_image, mask, batch_size*20, True)
    elif fill_method == 2:
        fill_image_with_avg(next_image, mask, batch_size*20, True)
    elif fill_method == 3:
        fill_image(next_image, mask, original_image)
    else :
        raise Exception("fill method dont exist")
    
    
    return next_image
    


def interpolate_flow(dps_images:np.ndarray, V:np.ndarray, timesDay, times, mask, batch_size:int=0, n:int=1, objects=[], show:bool=True):

    if objects == []:
        objects = range(V.shape[2])
      
    V_interpolation = np.zeros((V.shape[0], V.shape[1], n*len(objects)))
    timestampss = []
    mse = []
    l = 0   # l is a variable to increment every time an image is added to V_interpolation
    
    # interpolate flow
    for i, t in enumerate(objects):
        interpolate_images = []
        timestamps = []
        original_image = V[:,:,t]
        V_interpolation[:,:,l] = original_image
        l += 1
        #plt_imshow(original_image)
        for j in range(n-1):
            interpolate_image = predict(i, dps_images, original_image, j+1, n, batch_size, mask, 3)
            V_interpolation[:,:,l] = interpolate_image
            interpolate_images.append(interpolate_image)
            l += 1
        if t < V.shape[2]:
            mse.append(MSE(V[:,:,t + 1], interpolate_image, mask))
            
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
    return V_interpolation, (np.array(timestampss).ravel()).tolist(), np.mean(mse)



def move_pixel(img, img_origin, mask, source, target) -> None:
    """
    Move pixel from source to target
    ---
    Args:
        img: 2D array of image
        source: (x, y) coordinate of source pixel
        target: (x, y) coordinate of target pixel
    Return:
        Image with pixel moved
    """
    if not (target[0] >= img.shape[1] or target[1] >= img.shape[0] or target[0] < 0 or target[1] < 0):
        if mask[target[1], target[0]] == 1.0:
            if np.isnan(img[target[1], target[0]]):
                img[target[1], target[0]] = img_origin[source[1], source[0]]
                # print(f"Moving pixel from {source} to {target}, pixel value {img_origin[source[1], source[0]]}")
            # else:
            #     img[target[1], target[0]] = np.minimum(img[target[1], target[0]], img_origin[source[1], source[0]]) - np.abs(img[target[1], target[0]] - img_origin[source[1], source[0]])
    return

def extrapolate_flow(dps_images:np.ndarray, V:np.ndarray, timesDay, times, mask, minutes_after:int=15, batch_size:int=0, objects=[], show:bool=False) -> list:
    """
    extrapolate the flow between two images.
    ---
    Args:
        dps_images: Results of optical flow in the form of a 4D array (2, V.shape[0], V.shape[1], V.shape[2])
        V: 3D array of images after gaussian filtering (y, x, t)
        timesDay: List of dates
        times: List of times
        mask: Binary mask outlining Denmark
        base: Base image to extrapolate
        minutes_after: Minutes after the base image
        show: Whether to show the plot
    Return:
        List of extrapolated images
    """
    # interpolate flow
    extrapolate_images = []
    for i, t in enumerate(objects):
        original_image = V[:,:,t]
        # plt_imshow(original_image*mask)
        extrapolate_image = np.zeros_like(original_image).copy()
        extrapolate_image[:] = np.nan
        for y in range(dps_images.shape[1]):
            for x in range(dps_images.shape[2]):
                if (dps_images[0,y,x,i] == 0 or dps_images[1,y,x,i] == 0):
                    continue
                else:
                    # move pixel
                    dx = np.rint(dps_images[0,y,x,i]*(minutes_after)/(15)).astype(int)
                    dy = np.rint(dps_images[1,y,x,i]*(minutes_after)/(15)).astype(int)
                    # move pixels as a batch to same direction
                    if not (dx == 0 and dy == 0):
                        x_lower, x_upper = np.maximum(0, x-batch_size), np.minimum(V.shape[1], x+batch_size+1)
                        y_lower, y_upper = np.maximum(0, y-batch_size), np.minimum(V.shape[0], y+batch_size+1)
                        for y_batch in range(y_lower, y_upper):
                            for x_batch in range(x_lower, x_upper):
                                move_pixel(extrapolate_image, original_image, mask, (x_batch, y_batch), (x_batch+dx, y_batch+dy))
        # Fill nan values from the earth image
        fill_image(extrapolate_image, mask, V[:,:,t])
        extrapolate_images.append(extrapolate_image)
        
    timestamps = []
    for i, t in enumerate(objects):
        hour = int(times[t][:2])
        minutes = minutes_after + int(times[t][2:4])
        if minutes >= 60:
            minutes -= 60
            hour += 1
        timestamps.append(f"202403{timesDay[t]}_{str(hour).zfill(2)}{str(minutes).zfill(2)}{times[t][4:]}" )
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
    return extrapolate_images, timestamps

def extrapolate_flow_dont_work_yest(dps_images:np.ndarray, V:np.ndarray, timesDay, times, mask, minutes_after:int=15, batch_size:int=0, objects=[], show:bool=False) -> list:
    """
    extrapolate the flow between two images.
    ---
    Args:
        dps_images: Results of optical flow in the form of a 4D array (2, V.shape[0], V.shape[1], V.shape[2])
        V: 3D array of images after gaussian filtering (y, x, t)
        timesDay: List of dates
        times: List of times
        mask: Binary mask outlining Denmark
        base: Base image to extrapolate
        minutes_after: Minutes after the base image
        show: Whether to show the plot
    Return:
        List of extrapolated images
    """
    # interpolate flow
    extrapolate_images = []
    for i, t in enumerate(objects):
        original_image = V[:,:,t]

        extrapolate_image = np.zeros_like(original_image).copy()
        extrapolate_image[:] = np.nan
        
        predict(i, dps_images, V, t, j+1, n, batch_size, interpolate_images, mask, _)
        predict(i, dps_images, original_image, minutes_after, 15, batch_size, extrapolate_images, mask)
        
        # Fill nan values from the earth image
        fill_image(extrapolate_image, mask, original_image)
        extrapolate_images.append(extrapolate_image)
        
    timestamps = []
    for i, t in enumerate(objects):
        hour = int(times[t][:2])
        minutes = minutes_after + int(times[t][2:4])
        if minutes >= 60:
            minutes -= 60
            hour += 1
        timestamps.append(f"202403{timesDay[t]}_{str(hour).zfill(2)}{str(minutes).zfill(2)}{times[t][4:]}" )
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
    return extrapolate_images, timestamps


def MSE(y_true, y_pred, mask):
    return np.square(y_true[mask] - y_pred[mask]).sum()/mask.sum()

def plt_imshow(img):
    plt.imshow(img, cmap="viridis")
    plt.show()
    return

def fill_image(img, mask, background_image):
    for y in range(img.shape[0]):
        for x in range(img.shape[1]):
            if np.isnan(img[y,x]) and mask[y,x] == 1.0:
                img[y,x] = background_image[y,x]
            elif mask[y,x] == 0.0:
                img[y,x] = 0.0

def fill_image_with_avg(interpolate_image, mask, kernel_size=20, use_global_avg=False):
    # Ensure mask is boolean
    mask = mask.astype(bool)
    
    # Define the kernel
    kernel = np.ones((kernel_size, kernel_size))
    
    # Count valid (non-NaN) neighbors
    valid_mask = ~np.isnan(interpolate_image)
    valid_neighbor_count = ndimage.convolve(valid_mask.astype(float), kernel, mode='constant', cval=0)
    
    # Compute the sum of valid neighbors' values
    interpolated_values = ndimage.convolve(np.nan_to_num(interpolate_image), kernel, mode='constant', cval=0)
    
    # Calculate the average of the neighbors
    with np.errstate(invalid='ignore', divide='ignore'):  # Ignore division by zero or NaN results temporarily
        average_values = interpolated_values / valid_neighbor_count
    
    # Replace division by zero results (NaNs) with np.nan to ensure they can be filled later if global average is used
    average_values[np.isnan(average_values)] = np.nan
    
    # Identify where to place these averages: NaN locations within the mask
    fill_positions = np.isnan(interpolate_image) & mask
    
    # Fill these positions with the computed averages
    interpolate_image[fill_positions] = average_values[fill_positions]
    
    if use_global_avg:
        # Calculate the global average from the valid entries in the image
        global_avg = np.nanmean(interpolate_image)  # Use nanmean to ignore NaNs in global average calculation
        # Fill any remaining NaNs with the global average
        interpolate_image[np.isnan(interpolate_image)] = global_avg
    
    
    # Identify where to place these averages: NaN locations within the mask
    fill_positions = np.isnan(interpolate_image) & mask
    
    # Fill these positions with the computed averages
    interpolate_image[fill_positions] = average_values[fill_positions]
    
    if use_global_avg:
        global_avg = np.mean(interpolate_image[valid_mask])
        interpolate_image[np.isnan(average_values)] = global_avg


def fill_image_with_max(interpolate_image, mask, kernel_size=20, use_global_max=False):

    # Find the maximum among the valid neighbors
    max_values = ndimage.maximum_filter(interpolate_image, size=kernel_size, mode='constant', cval=np.nan)
    
    # Identify where to place these max values: NaN locations within the mask
    fill_positions = np.isnan(interpolate_image) & mask
    
    # Fill these positions with the computed max values
    interpolate_image[fill_positions] = max_values[fill_positions]
    
    if use_global_max and np.isnan(interpolate_image).any(): 
        # Calculate the global maximum from the valid entries in the image
        global_max = np.nanmax(interpolate_image)
        # Fill any remaining NaNs with the global maximum
        interpolate_image[np.isnan(interpolate_image)] = global_max
    



if __name__ == "__main__":
    path = 'Project4/Processedfull'
    files_in_directory = os.listdir(path)
    target_days = [file.replace('.xlsx','') for file in files_in_directory if file.endswith('.xlsx')]
    mse_list = []
    for target in target_days:
        V, timesDay, times, mask = load_images(target, path)
        print(timesDay, times)
        # Define how many objects subject to calculating optical flow
        objects=range(5)
        dps_images = Lucas_Kanade_method(V, objects=objects)
        # plot_with_noise_filtering(dps_images, V, timesDay, times, mask, show=False)

        _, _, mse = interpolate_flow(dps_images, V, timesDay, times, mask, objects=objects, show=True, n=3)
        mse_list.append(mse)
        # extrapolate_flow(dps_images, V, timesDay, times, mask, minutes_after=15, objects=objects, show=True)
    mse = np.mean(mse_list)