import pandas as pd

data = pd.read_csv('D:/Mines/3A/ML/Projet/archive/Radar_Traffic_Counts.csv')

data = data.drop(columns=['Time Bin'])
data['Direction'] = data['Direction'].astype('category').cat.codes

col = ['location_name', 'location_latitude', 'location_longitude', 'Year', 'Month', 'Day', 'Day of Week', 'Hour', 'Direction']
data = data.groupby(col)['Volume'].sum().reset_index()

data = data.pivot_table(index=['location_name', 'location_latitude', 'location_longitude', 'Year', 'Month', 'Day', 'Day of Week', 'Direction'], columns='Hour', values='Volume').reset_index()
#print(data[(data['location_latitude'] <= 30.402286) & (data['Year'] == 2017) & (data['Month'] == 9) & (data['Direction'] == 1)])
#data = data.interpolate('time')

data.interpolate(method='linear', inplace=True) # après sa il ne reste que 2 lignes comprenant des valeurs NaN dans leur séries; nous allons les supprimer
data = data.dropna()
print(data[:20])
