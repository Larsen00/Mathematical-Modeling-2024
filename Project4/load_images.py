import glob
import numpy as np
import os


def load_images(target:str, path:str) -> np.ndarray:
    """
    Load images from the given dates.
    ---
    Args:
        target: List of dates to load images from
        path: Path to the directory containing the images
    Return:
        V: Images
        timesDay: Dates
        times: time of day
        mask: boolean mask of danish soil
    """
    # Allocate memory and load image data
    times = []
    timesDay = []
    ims = []

    # Loading dates from file names
    files_in_directory = os.listdir(path)
    dates = [file.replace('.xlsx','') for file in files_in_directory if file.endswith('.xlsx')]

    # Find all image files
    file_name = []
    for day in dates:
        if target in day:
            file_name += glob.glob(f'{path}/{day[6:8]}/*natural_color.npy')
    
    # Sort the file names in time
    ind = file_name[0].find('202403')
    file_name.sort(key=lambda x: int(x[ind+12:ind+14]))
    file_name.sort(key=lambda x: int(x[ind+10:ind+12]))
    file_name.sort(key=lambda x: int(x[ind+8:ind+10]))

    if len(file_name) == 0:
        raise ValueError("No images found.")

    # Load binary mask outlining Denmark
    mask = np.load(f'{path}/mask.npy') 
    for i, entry in enumerate(file_name):
        img = np.load(entry)
        dummy = img[:,:,0]+img[:,:,1]-2*img[:,:,2]

        # Find time information in filename
        ind = entry.find('202403')
        
        times.append(entry[ind+8:ind+14])   # gives time of day hhmmss
        timesDay.append(entry[ind+6:ind+8]) # gives the date
        
        # read images
        ims.append(dummy)

        # NOTE Does it make sense to make optical between days? dates.index(entry[ind+6:ind+8])
        # print(f"Progress: {(i+1)/len(file_name)*100:.2f}%", end="\r")
        
    timesDay = np.array(timesDay)
    times = np.array(times)

    # make image array (y,x,t)
    V = np.dstack(ims)
    print(f"Reading images done! {len(ims)} images read.")
    timesDay = np.array(timesDay)
    times = np.array(times)
    return V, timesDay, times, mask

if __name__ == "__main__":
    path = 'Project4/Processedfull'
    target_days = ['0317']
    for target in target_days:
        V, timesDay, times, mask = load_images(target, path)
        pass