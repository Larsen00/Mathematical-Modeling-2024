# earth image removed
import numpy as np
import scipy.ndimage
import matplotlib.pyplot as plt
import os
from load_images import load_images

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

def interpolate_flow(dps_images:np.ndarray, V:np.ndarray, timesDay, times, mask, batch_size:int=0, n:int=1, objects=[], show:bool=True):
    """
    Interpolate the flow between two images.
    ---
    Args:
        dps_images: Results of optical flow in the form of a 4D array (2, V.shape[0], V.shape[1], V.shape[2])
        V: 3D array of images after gaussian filtering (y, x, t)
        timesDay: List of dates
        times: List of times
        mask: Binary mask outlining Denmark
        n: Number of interpolated images(include last image)
        objects: List of objects to interpolate
        show: Whether to show the plot
    Return:
        return interpolated images from based image to the last interpolated image, in one 3D array.
    """
    
    # if show == False:
    #     make_dir(f"{path}/_interpolated")

    if objects == []:
        objects = range(V.shape[2])

    # interpolate flow
    # interpolate_imagess = []
    V_interpolation = np.zeros((V.shape[0], V.shape[1], n*len(objects)))
    timestampss = []
    l = 0   # l is a variable to increment every time an image is added to V_interpolation
    for i, t in enumerate(objects):
        interpolate_images = []
        timestamps = []
        original_image = V[:,:,t]
        V_interpolation[:,:,l] = original_image
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

def MSE(y_true, y_pred, mask):
    return np.square(y_true[mask] - y_pred[mask]).sum()/(mask.sum())

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

if __name__ == "__main__":
    path = 'Project4/Processedfull'
    target_days = ['0317']
    for target in target_days:
        V, timesDay, times, mask = load_images(target, path)
        print(timesDay, times)
        # Define how many objects subject to calculating optical flow
        objects=range(5)
        dps_images = Lucas_Kanade_method(V, objects=objects)
        # plot_with_noise_filtering(dps_images, V, timesDay, times, mask, show=False)

        # interpolate_flow(dps_images, V, timesDay, times, mask, objects=objects, show=True, n=3)
        # extrapolate_flow(dps_images, V, timesDay, times, mask, minutes_after=15, objects=objects, show=True)