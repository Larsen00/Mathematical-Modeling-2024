from peaksolarprod import *
from flow_ver3 import *
from datetime import datetime
from ground import extract_groundintensity

path = "Project4/Processedfull"
target_days = ['0317']
objects=range(10,20) #Which frames to predict
avg_distances = []
errors = []
pred_im = []
earth_im = extract_groundintensity()
for i in range(len(target_days)):
        excel_data = pd.read_excel(f'Project4/Processedfull/2024{target_days[i]}.xlsx', usecols="A,F") 
        target_times = pd.to_datetime(excel_data['Minutes1UTC']).dt.strftime('%H%M%S')
        V, timesDay, times_predicted, mask = load_images(target_days[i], path)

        # Define how many objects subject to calculating optical flow
        dps_images = Lucas_Kanade_method(V, objects=objects)

        # plot_with_noise_filtering(dps_images, V, timesDay, times, mask, show=False)
        # interpolate_flow(dps_images, V, timesDay, times, mask, objects=objects, show=True)
        pred_ims,timestamp = extrapolate_flow(dps_images, V,timesDay, times_predicted, mask, minutes_after=15, objects=objects, show=False)
       # Assuming pred_im is already defined as per your description
        pred_ims = np.hstack([arr.reshape(-1, 1) for arr in pred_ims])  # Reshape each array to (57*156, 1)
  
        # Apply the mask to each column
        mask = mask.flatten()
        masked_ims = pred_ims[mask, :]
        y_true = []
        for i, frame_index in enumerate(objects):
            time = times_predicted[frame_index]
            minutes = float(time[2:4])/60
            hour = float(time[0:2])+minutes
            hour_factor = hour_func(hour)
            masked_ims[:,i] = masked_ims[:,i]*hour_factor

            timestamp = time[:4]+'00'
            # Correct filtering approach using Pandas
            condition = target_times == timestamp
            matching_indices = target_times[condition].index.tolist()
            if matching_indices:
                matching_row = excel_data.iloc[matching_indices]
                y_true.append(matching_row['SolarPower'].values[0])
        
        y_pred = model.predict(masked_ims.T)

        #Get correct production values
        #excelfile = f'{timestamp}.xlsx'
        errors.append(((np.mean((y_true-y_pred)**2))))
        avg_distances.append(np.mean((y_true-y_pred)))
print(np.mean(errors))
print(np.mean(avg_distances))
print('end')