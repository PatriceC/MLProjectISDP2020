import pandas as pd
import numpy as np

import torch

import random
import time

def series(Date_J, latitude, longitude, direction, longueur_serie, data):
    """
        Retourne 3 séries de longueur_serie valeurs de Volume pour un jour, une position, et une direction
    """
    row_J = data[(np.isclose(data['location_latitude'], latitude, rtol=1.e-4, atol=1.e-6)) & (np.isclose(data['location_longitude'], longitude, rtol=1.e-4, atol=1.e-6)) & (data['Date'] == Date_J) & (data['Direction'] == direction)]
    row_J_moins_1 = data[(np.isclose(data['location_latitude'], latitude, rtol=1.e-4, atol=1.e-6)) & (np.isclose(data['location_longitude'], longitude, rtol=1.e-4, atol=1.e-6)) & (data['Date'] == Date_J - pd.to_timedelta(1, unit='d')
) & (data['Direction'] == direction)]
    row_J_moins_2 = data[(np.isclose(data['location_latitude'], latitude, rtol=1.e-4, atol=1.e-6)) & (np.isclose(data['location_longitude'], longitude, rtol=1.e-4, atol=1.e-6)) & (data['Date'] == Date_J - pd.to_timedelta(2, unit='d')
) & (data['Direction'] == direction)]
    row_J_moins_7 = data[(np.isclose(data['location_latitude'], latitude, rtol=1.e-4, atol=1.e-6)) & (np.isclose(data['location_longitude'], longitude, rtol=1.e-4, atol=1.e-6)) & (data['Date'] == Date_J - pd.to_timedelta(7, unit='d')
) & (data['Direction'] == direction)]
    row_J_moins_8 = data[(np.isclose(data['location_latitude'], latitude, rtol=1.e-4, atol=1.e-6)) & (np.isclose(data['location_longitude'], longitude, rtol=1.e-4, atol=1.e-6)) & (data['Date'] == Date_J - pd.to_timedelta(8, unit='d')
) & (data['Direction'] == direction)]

    if (row_J.empty or row_J_moins_1.empty or row_J_moins_7.empty):
        return(None)
    
    elif (row_J_moins_2.empty or row_J_moins_8.empty):
        valeurs_J, valeurs_J_moins_1, valeurs_J_moins_7 = row_J.values.tolist()[0][8:], row_J_moins_1.values.tolist()[0][8:], row_J_moins_7.values.tolist()[0][8:]

        heure_min = longueur_serie - 1
        nb_series = 24 - 2 * heure_min
        target, serie_J, serie_J_moins_1, serie_J_moins_7 = np.zeros(nb_series), np.zeros((nb_series,longueur_serie-1)), np.zeros((nb_series,longueur_serie)), np.zeros((nb_series,longueur_serie))

        for h in range(nb_series):
            target[h] = valeurs_J[heure_min + h + longueur_serie - 1]
            serie_J[h] = valeurs_J[heure_min + h : heure_min + h + longueur_serie - 1]
            serie_J_moins_1[h] = valeurs_J_moins_1[heure_min + h : heure_min + h + longueur_serie]
            serie_J_moins_7[h] = valeurs_J_moins_7[heure_min + h : heure_min + h + longueur_serie]

    else:
        valeurs_J, valeurs_J_moins_1, valeurs_J_moins_2, valeurs_J_moins_7, valeurs_J_moins_8 = row_J.values.tolist()[0][8:], row_J_moins_1.values.tolist()[0][8:], row_J_moins_2.values.tolist()[0][8:], row_J_moins_7.values.tolist()[0][8:], row_J_moins_8.values.tolist()[0][8:]
        V_J, V_J_1, V_J_7 = valeurs_J + valeurs_J_moins_1, valeurs_J_moins_1 + valeurs_J_moins_2, valeurs_J_moins_7 + valeurs_J_moins_8

        heure_min = 24
        nb_series = 24 - (longueur_serie - 1)
        target, serie_J, serie_J_moins_1, serie_J_moins_7 = np.zeros(nb_series), np.zeros((nb_series,longueur_serie-1)), np.zeros((nb_series,longueur_serie)), np.zeros((nb_series,longueur_serie))

        for h in range(nb_series):
            target[h] = V_J[heure_min + h + longueur_serie - 1]
            serie_J[h] = V_J[heure_min + h : heure_min + h + longueur_serie - 1]
            serie_J_moins_1[h] = V_J_1[heure_min + h : heure_min + h + longueur_serie]
            serie_J_moins_7[h] = V_J_7[heure_min + h : heure_min + h + longueur_serie]
    
    return(target, serie_J, serie_J_moins_1, serie_J_moins_7)

def process_data(longueur_serie=6, file='./Radar_Traffic_Counts.csv'):
    data = pd.read_csv(file)

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
    # On garde les valeurs de mois entre 0 et 11 (plutôt que 1 et 12), ce qui sera plus pratique pour créer des one-hot vectors
    data['Month'] = data['Month'] - 1

    data_train, data_test = [], []
    for _, row in data.iterrows():
        latitude, longitude = row['location_latitude'], row['location_longitude']
        month, day_week, date = row['Month'], row['Day of Week'], row['Date']
        direction = row['Direction']

        result = series(Date_J=date, latitude=latitude, longitude=longitude, direction=direction, longueur_serie=longueur_serie, data=data)

        if result is not None:
            target, serie_J, serie_J_moins_1, serie_J_moins_7 = result

            for t, s1, s2, s3 in zip(target, serie_J, serie_J_moins_1, serie_J_moins_7):
                if random.random() < 0.9:
                    data_train.append([latitude, longitude, month, day_week, direction] + s1.tolist() + s2.tolist() + s3.tolist() + [t])
                else:
                    data_test.append([latitude, longitude, month, day_week, direction] + s1.tolist() + s2.tolist() + s3.tolist() + [t])

    random.shuffle(data_train)
    random.shuffle(data_test)

    np.savetxt('./data_train_' + str(longueur_serie) + '.txt', np.array(data_train))
    np.savetxt('./data_test_' + str(longueur_serie) + '.txt', np.array(data_test))

    return(np.array(data_train), np.array(data_train))