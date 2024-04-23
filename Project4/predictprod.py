from peaksolarprod import *
from flow_ver2 import *
path = "./Project4"

pred_im = []
target_days = ['0317']
objects=[15]
avg_distances = []
errosr = []
for i in range(len(objects)):
        V, timesDay, times, mask = load_images(target_days[i], path)

        # Define how many objects subject to calculating optical flow
        dps_images = Lucas_Kanade_method(V, objects=objects[i])

        # plot_with_noise_filtering(dps_images, V, timesDay, times, mask, show=False)
        # interpolate_flow(dps_images, V, timesDay, times, mask, objects=objects, show=True)
        pred_im,timestamp= extrapolate_flow(dps_images, V, timesDay, times, mask, minutes_after=15, objects=objects[i], show=True)
        pred_im = pred_im[0][mask].flatten()
        if np.std(pred_im) != 0:
            pred_im = (pred_im-np.mean(dummy))/np.std(pred_im)

        #Write code to calculate month and hour of predicted image. 
        #month_factor = month_func(month_new)
        #hour_factor = hour_func(hour_new,month_new)
        futureimage = pred_im*month_factor*hour_factor
        y_pred = model.predict(futureimage)

        #Get correct production values
        #excelfile = f'{timestamp}.xlsx'
        excel_data = pd.read_excel(excel_file, usecols="A,F") 

        matching_row = excel_data[excel_data['Minutes1UTC'][-6,:]==timestamp]
        y_true = matching_row['solar_power']

        error = ((np.mean((y_true-y_true)**2)))
        avg_dist = np.mean(np.abs(y_true-y_true))
        errors.append(error)
        avg_distances.append(avg_dist)
print(errors)
print(avg_distances)
