from peaksolarprod import *
from flow2 import futureimage,hour_new,month_new,timestamp

month_factor = month_func(month_new)
hour_factor = hour_func(hour_new,month_new)
futureimage = futureimage*month_factor*hour_factor
ydata = model.predict(futureimage)
excelfile = f'{timestamp}.xlsx'
excel_data = pd.read_excel(excel_file, usecols="A,F") 

matching_row = excel_data[excel_data['Minutes1UTC'][-6,:]==timestamp]
y_true = matching_row['solar_power']