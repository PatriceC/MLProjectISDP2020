import pandas as pd

data = pd.read_csv('D:/Mines/3A/ML/Projet/archive/Radar_Traffic_Counts.csv')

data = data.drop(columns=['Time Bin'])
data['Direction'] = data['Direction'].astype('category').cat.codes

col = ['location_name', 'location_latitude', 'location_longitude', 'Year', 'Month', 'Day', 'Day of Week', 'Hour', 'Direction']
data = data.groupby(col)['Volume'].sum().reset_index()
print(data.head())
data = data.pivot_table(index=['location_name', 'location_latitude', 'location_longitude', 'Year', 'Month', 'Day', 'Day of Week', 'Direction'], columns='Hour', values='Volume', fill_value=0).reset_index()
print(data.head())
print(data[0,:])
