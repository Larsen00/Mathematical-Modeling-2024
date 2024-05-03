import matplotlib.pyplot as plt
import matplotlib.dates as mdates 
from datetime import datetime
import numpy as np

errors = np.load('errors.npy', allow_pickle=True)
average_distances = np.load('average_distances.npy', allow_pickle=True)
y_prediction_list = np.load('y_prediction_list.npy', allow_pickle=True)
y_true_list = np.load('y_true_list.npy', allow_pickle=True)
true_time_list = np.load('true_time_list.npy', allow_pickle=True)
errors_interpolation = np.load('errors_interpolation.npy', allow_pickle=True)
average_distances_interpolation = np.load('average_distances_interpolation.npy', allow_pickle=True)
y_prediction_list_interpolation = np.load('y_prediction_list_interpolation.npy', allow_pickle=True)

test_date = "20240331"
x = [datetime.strptime(test_date + t, "%Y%m%d%H%M%S") for t in true_time_list]
plt.plot(x, y_prediction_list, label='Prediction')
plt.plot(x, y_prediction_list_interpolation, label='Prediction with interpolation')
plt.plot(x, y_true_list, label='true values')
plt.legend()
plt.title('Solar production prediction')
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
plt.gcf().autofmt_xdate()
plt.tight_layout()
plt.savefig('line3.svg')
plt.clf()

# print(np.mean(errors))
# print(np.mean(errors_interpolation))
plt.plot(x, errors, label='Prediction without interplation')
plt.plot(x, errors_interpolation, label='Prediction with interpolation')
print(f"MSE Error: {np.mean(errors)}")
print(f"MSE Error with interpolation: {np.mean(errors_interpolation)}")
plt.legend()
plt.title('MSE')
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
plt.gcf().autofmt_xdate()
plt.yscale('log')
plt.tight_layout()
plt.savefig('line4.svg')
plt.clf()

plt.plot(x, average_distances,'o', label='Prediction without interpolation')
plt.plot(x, average_distances_interpolation, 'o',label='Prediction with interpolation')
print(f"Average Distance: {np.mean(average_distances)}")
print(f"Average Distance with interpolation: {np.mean(average_distances_interpolation)}")
plt.axhline(y=0.0, color='r', linestyle='--')
plt.title('Residuals')
plt.legend()
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
plt.gcf().autofmt_xdate()
plt.tight_layout()
plt.savefig('line5.svg')
plt.clf()

x = [datetime.strptime(test_date + t, "%Y%m%d%H%M%S") for t in true_time_list]
plt.plot(y_true_list, y_prediction_list, 'o', label='Prediction without interpolation')
plt.plot(y_true_list, y_prediction_list_interpolation, 'o', label='Prediction with interpolation')
a = np.arange(500)
plt.plot(a, a, label='y=x')
plt.xlabel('True values')
plt.ylabel('Predicted values')
plt.legend()
plt.tight_layout()
plt.savefig('line6.svg')