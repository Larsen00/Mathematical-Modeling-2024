from peaksolarprod import *
from flow_ver2 import *
from ground import extract_groundintensity
from datetime import datetime
path = "./Project4/Processedfull"

pred_im = []
target_days = ['0317','0318']
objects=range(20,40)
avg_distances = []
errors = []
earth_image = extract_groundintensity()
for i in range(len(target_days)):
        V, timesDay, times, mask = load_images(target_days[i], path)

        # Define how many objects subject to calculating optical flow
        dps_images = Lucas_Kanade_method(V, objects=objects)

        # plot_with_noise_filtering(dps_images, V, timesDay, times, mask, show=False)
        # interpolate_flow(dps_images, V, timesDay, times, mask, objects=objects, show=True)
        pred_im,timestamp = extrapolate_flow(dps_images, V, earth_image,timesDay, times, mask, minutes_after=15, objects=objects, show=False)
       # Assuming pred_im is already defined as per your description
        reshaped_arrays = [arr.reshape(-1, 1) for arr in pred_im]  # Reshape each array to (57*156, 1)
        final_array = np.hstack(reshaped_arrays)  # Stack them horizontally
        # Apply the mask to each column
        mask = mask.flatten()
        masked_data = final_array[mask, :]

        #if np.std(final_array,axis=0) != 0:
        masked_data = (masked_data-np.mean(dummy,axis=0))/np.std(masked_data,axis=0)

        #Write code to calculate month and hour of predicted image. 
        #month_factor = month_func(month_new)
      

        #fix timestamp
            # Format the timestamp properly
        timestamp = datetime.strptime(timestamp, '%Y%m%d_%H%M%S').strftime('%H%M%S')
        timestamp = timestamp[:4] + '00'
        if int(timestamp[2:4]) > 30:
            hour_new = int(timestamp[0:2])+1
        else: hour_new = int(timestamp[0:2])
        hour_factor = hour_func(hour_new)
        futureimage = masked_data*month_factor*hour_factor
        y_pred = model.predict(futureimage.T)

        #Get correct production values
        #excelfile = f'{timestamp}.xlsx'
        excel_data = pd.read_excel(excel_file, usecols="A,F") 

        target_times = pd.to_datetime(excel_data['Minutes1UTC']).dt.strftime('%H%M%S')

        # Correct filtering approach using Pandas
        condition = target_times == timestamp
        matching_indices = target_times[condition].index.tolist()
        if matching_indices:
            matching_row = excel_data.iloc[matching_indices]
            y_true = matching_row['SolarPower'].values
        

        error = ((np.mean((y_true-y_pred)**2)))
        avg_dist = np.mean(np.abs(y_true-y_pred))
        errors.append(error)
        avg_distances.append(avg_dist)
print(errors)
print(avg_distances)
