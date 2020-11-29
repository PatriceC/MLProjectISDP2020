import pandas as pd
import numpy as np

data = pd.read_csv('D:/Mines/3A/ML/Projet/archive/Radar_Traffic_Counts.csv')

data = data.drop(columns=['location_name', 'Time Bin'])
data['Direction'] = data['Direction'].astype('category').cat.codes
data['Date'] = pd.to_datetime(data[['Year', 'Month', 'Day']], errors = 'coerce')

col = ['location_latitude', 'location_longitude', 'Year', 'Month', 'Day', 'Date', 'Day of Week', 'Hour', 'Direction']
col_no_hour = ['location_latitude', 'location_longitude', 'Year', 'Month', 'Day', 'Date', 'Day of Week', 'Direction']
data = data.groupby(col)['Volume'].sum().reset_index()
data = data.pivot_table(index=col_no_hour, columns='Hour', values='Volume').reset_index()

data.interpolate(method='linear', inplace=True) # Après ça, il ne reste que 2 lignes comprenant des valeurs NaN dans leurs séries; nous allons les supprimer
data = data.dropna()

# On normalise (méthode min-max) les valeurs de latitude et longitude
data['location_latitude'] = (data['location_latitude'] - data['location_latitude'].min()) / (data['location_latitude'].max() - data['location_latitude'].min())
data['location_longitude'] = (data['location_longitude'] - data['location_longitude'].min()) / (data['location_longitude'].max() - data['location_longitude'].min())
print(data[2500:2505])

#print(data[np.isclose(data['location_latitude'], latitude, rtol=1.e-4, atol=1.e-6)])

def series(heure, Date_J, latitude, longitude, direction, longueur_serie):
    """
        Retourne 3 séries de valeurs de Volume pour une heure donnée, un jour, une position, et une direction
        longueur_serie : 
    """
    serie_J, serie_J_moins_1, serie_J_moins_7 = [], [], []
    if heure >= (longueur_serie - 1):
        row_J = data[(np.isclose(data['location_latitude'], latitude, rtol=1.e-4, atol=1.e-6)) & (np.isclose(data['location_longitude'], longitude, rtol=1.e-4, atol=1.e-6)) & (data['Date'] == Date_J) & (data['Direction'] == direction)]
        row_J_moins_1 = data[(np.isclose(data['location_latitude'], latitude, rtol=1.e-4, atol=1.e-6)) & (np.isclose(data['location_longitude'], longitude, rtol=1.e-4, atol=1.e-6)) & (data['Date'] == Date_J - pd.to_timedelta(1, unit='d')
) & (data['Direction'] == direction)]
        row_J_moins_7 = data[(np.isclose(data['location_latitude'], latitude, rtol=1.e-4, atol=1.e-6)) & (np.isclose(data['location_longitude'], longitude, rtol=1.e-4, atol=1.e-6)) & (data['Date'] == Date_J - pd.to_timedelta(7, unit='d')
) & (data['Direction'] == direction)]

        for h in range(heure + 1 - longueur_serie, heure + 1):
            if h != heure:
                serie_J.append(row_J[h].values[0])
            serie_J_moins_1.append(row_J_moins_1[h].values[0])
            serie_J_moins_7.append(row_J_moins_7[h].values[0])
    

    return(serie_J, serie_J_moins_1, serie_J_moins_7)


a, b, c = series(17, pd.to_datetime('2018-06-03'), 0.02498, 0.639365, 1, 4)

print(a,b,c)

"""
"""