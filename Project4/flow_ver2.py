import glob
import numpy as np
import scipy.ndimage
import matplotlib.pyplot as plt
import os
from load_images import load_images
from ground import extract_groundintensity

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
                print(f"Progress: {(i*V.shape[0]*V.shape[1] + y*V.shape[1] + x)/total_volume*100:.2f}%", end="\r")
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

def interpolate_flow(dps_images:np.ndarray, V:np.ndarray, earth_image:np.ndarray, timesDay, times, mask, batch_size:int=0, n:int=1, objects=[], show:bool=True):
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
    """
    
    if show == False:
        make_dir(f"{path}/_interpolated")

    if objects == []:
        objects = range(V.shape[2])

    # interpolate flow
    for i, t in enumerate(objects):
        interpolate_images = []
        original_image = V[:,:,t]
        interpolate_image = np.zeros_like(original_image).copy()
        interpolate_image[:] = np.nan
        for j in range(n):
            for y in range(dps_images.shape[1]):
                for x in range(dps_images.shape[2]):
                    if (dps_images[0,y,x,i] == 0 or dps_images[1,y,x,i] == 0):
                        continue
                    else:
                        # move pixel
                        dx = np.rint(dps_images[0,y,x,i]*(j+1)/(n)).astype(int)
                        dy = np.rint(dps_images[1,y,x,i]*(j+1)/(n)).astype(int)
                        # move pixels as a batch to same direction
                        if dx != 0 and dy != 0:
                            x_lower, x_upper = np.maximum(0, x-batch_size), np.minimum(V.shape[1], x+batch_size+1)
                            y_lower, y_upper = np.maximum(0, y-batch_size), np.minimum(V.shape[0], y+batch_size+1)
                            for y_batch in range(y_lower, y_upper):
                                for x_batch in range(x_lower, x_upper):
                                    move_pixel(interpolate_image, original_image, mask, (x_batch, y_batch), (x_batch+dx, y_batch+dy))
            # Fill nan values from the earth image
            fill_earth_image(interpolate_image, mask)
            interpolate_images.append(interpolate_image)
        
        # plot_images consists of [start image,...interpolated images..., end image]
        plot_images = [V[:,:,t]] + interpolate_images + [V[:,:,t+1]]
      
        # # Take mask on every images
        # for i in range(len(plot_images)):
        #     plot_images[i] = plot_images[i]*mask
        
        fig, ax = plt.subplots(nrows=1,ncols=3, figsize=(16,9))
        ax[0].imshow(plot_images[-2], cmap="viridis")
        ax[0].set_title(f"Interpolated Image")
        ax[1].imshow(plot_images[-1], cmap="viridis")
        ax[1].set_title(f"Real Image")
        diff = plot_images[-2] - plot_images[-1]
        ax[2].imshow(diff, cmap="viridis")
        ax[2].set_title(f"Difference MSE: {np.square(diff).sum()/len((mask == 1.0).ravel()):.2f} Baseline MSE: {np.square((V[:,:,t]-V[:,:,t+1])*mask).sum()/len((mask == 1.0).ravel()):.2f}")
        fig.suptitle(f"Interpolated Flow 202403{timesDay[t+1]}_{times[t+1]}")
        fig.tight_layout()
        if show:
            plt.show()
        else:
            plt.savefig(f"{path}/_interpolated/interpolated_flow_202403{timesDay[t]}_{times[t]}.png")
    return interpolate_images

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
                print(f"Moving pixel from {source} to {target}")
            else:
                img[target[1], target[0]] = np.minimum(img[target[1], target[0]], img_origin[source[1], source[0]]) - np.abs(img[target[1], target[0]] - img_origin[source[1], source[0]])

    # Make sure that the target pixel is within the image
    # if not (target[0] >= img.shape[1] or target[1] >= img.shape[0] or target[0] < 0 or target[1] < 0):
    #     # Move pixel from source to target
    #     if np.isnan(img[target[1], target[0]]) and mask[target[1], target[0]] == 1.0:
    #         val = img_origin[source[1], source[0]]
    #     else:
    #         val = np.minimum(img[target[1], target[0]], img_origin[source[1], source[0]]) - np.abs(img[target[1], target[0]] - img_origin[source[1], source[0]])
    #     # give value to target pixel
    #     img[target[1], target[0]] = val
    #     print(f"Moving pixel from {source} to {target}, pixel value {val}")
    # print(source, target)

    # # Move pixel from the opposite direction of the target to the source to the source.
    # source = (2*source[0]-target[0], 2*source[1]-target[1])
    # target = ((source[0] + target[0])//2, (source[1] + target[1])//2)
    # # print(source, target)
    # if not (source[0] >= img.shape[1] or source[1] >= img.shape[0] or source[0] < 0 or source[1] < 0):
    #     img[target[1], target[0]] = img_origin[source[1], source[0]]
    #     print(f"Moving pixel from {source} to {target}")
    return

def extrapolate_flow(dps_images:np.ndarray, V:np.ndarray, earth_image:np.ndarray, timesDay, times, mask, minutes_after:int=15, batch_size:int=0, objects=[], show:bool=False) -> list:
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
        for y in range(dps_images.shape[1]):
            for x in range(dps_images.shape[2]):
                if (dps_images[0,y,x,i] == 0 or dps_images[1,y,x,i] == 0):
                    continue
                else:
                    # move pixel
                    dx = np.rint(dps_images[0,y,x,i]*(minutes_after)/(15)).astype(int)
                    dy = np.rint(dps_images[1,y,x,i]*(minutes_after)/(15)).astype(int)
                    # move pixels as a batch to same direction
                    if dx != 0 and dy != 0:
                        x_lower, x_upper = np.maximum(0, x-batch_size), np.minimum(V.shape[1], x+batch_size+1)
                        y_lower, y_upper = np.maximum(0, y-batch_size), np.minimum(V.shape[0], y+batch_size+1)
                        for y_batch in range(y_lower, y_upper):
                            for x_batch in range(x_lower, x_upper):
                                move_pixel(extrapolate_image, original_image, mask, (x_batch, y_batch), (x_batch+dx, y_batch+dy))
        # Fill nan values from the earth image
        fill_earth_image(extrapolate_image, mask)
        extrapolate_images.append(extrapolate_image)
    if show:
        for i, t in enumerate(objects):
            plt.imshow(extrapolate_images[i]*mask, cmap="viridis")
            plt.title(f"Extrapolated Image {minutes_after} minutes after 202403{timesDay[t]}_{times[t]}")
            plt.tight_layout()
            plt.show()
    
    return extrapolate_images, f"202403{timesDay[t]}_{times[t][:2]}{str(minutes_after + int(times[t][2:4])).zfill(2)}{times[t][4:]}"

def fill_earth_image(img, mask):
    for y in range(img.shape[0]):
        for x in range(img.shape[1]):
            if np.isnan(img[y,x]) and mask[y,x] == 1.0:
                img[y,x] = earth_image[y,x]
            elif mask[y,x] == 0.0:
                img[y,x] = 0.0

if __name__ == "__main__":
    path = 'Project4/Processedfull'
    target_days = ['0320']
    for target in target_days:
        V, timesDay, times, mask = load_images(target, path)
        earth_image = extract_groundintensity()
        print(timesDay, times)
        # Define how many objects subject to calculating optical flow
        objects=range(20,25)
        dps_images = Lucas_Kanade_method(V, objects=objects)


        # plot_with_noise_filtering(dps_images, V, timesDay, times, mask, show=False)
        # print(len(interpolate_flow(dps_images, V, earth_image, timesDay, times, mask, objects=objects, show=True, n=15)))
        print(extrapolate_flow(dps_images, V, earth_image, timesDay, times, mask, minutes_after=15, objects=objects, show=True))
