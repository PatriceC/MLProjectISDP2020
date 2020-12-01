# -*- coding: utf-8 -*-
"""
Created on Mon Nov 30 16:38:39 2020

@author: Patrice CHANOL & Corentin MORVAN--CHAUMEIL
"""

import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt

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

def process_data(date_range=['2017','2020'], direction=None, latitude=[-100,100], longitude=[-100,100], longueur_serie=6, file='./Radar_Traffic_Counts.csv'):
    data = pd.read_csv(file)

    data = data.drop(columns=['location_name', 'Time Bin'])
    data['Direction'] = data['Direction'].astype('category').cat.codes
    data['Date'] = pd.to_datetime(data[['Year', 'Month', 'Day']], errors = 'coerce')

    data = data.sort_values(['Year', 'Month', 'Day'])

    data = data[(data['Date'] >= date_range[0]) & (data['Date'] <= date_range[1])]
    data = data[(data['location_latitude'] >= latitude[0]) & (data['location_latitude'] <= latitude[1])]
    data = data[(data['location_longitude'] >= longitude[0]) & (data['location_longitude'] <= longitude[1])]

    if direction != None:
            data = data[data['Direction'] == direction]

    col = ['location_latitude', 'location_longitude', 'Year', 'Month', 'Day', 'Date', 'Day of Week', 'Hour', 'Direction']
    col_no_hour = ['location_latitude', 'location_longitude', 'Year', 'Month', 'Day', 'Date', 'Day of Week', 'Direction']
    data = data.groupby(col)['Volume'].sum().reset_index()
    data = data.pivot_table(index=col_no_hour, columns='Hour', values='Volume').reset_index()

    #data.interpolate(method='linear', inplace=True) # Après ça, il ne reste que 2 lignes comprenant des valeurs NaN dans leurs séries; nous allons les supprimer
    data = data.dropna()

    # On normalise (méthode min-max) les valeurs de latitude et longitude
    data['location_latitude'] = (data['location_latitude'] - data['location_latitude'].min()) / (data['location_latitude'].max() - data['location_latitude'].min())
    data['location_longitude'] = (data['location_longitude'] - data['location_longitude'].min()) / (data['location_longitude'].max() - data['location_longitude'].min())
    # On garde les valeurs de mois entre 0 et 11 (plutôt que 1 et 12), ce qui sera plus pratique pour créer des one-hot vectors
    data['Month'] = data['Month'] - 1

    data_post = []
    for _, row in data.iterrows():
        latitude, longitude = row['location_latitude'], row['location_longitude']
        month, day_week, date = row['Month'], row['Day of Week'], row['Date']
        direction = row['Direction']

        result = series(Date_J=date, latitude=latitude, longitude=longitude, direction=direction, longueur_serie=longueur_serie, data=data)

        if result is not None:
            target, serie_J, serie_J_moins_1, serie_J_moins_7 = result

            for t, s1, s2, s3 in zip(target, serie_J, serie_J_moins_1, serie_J_moins_7):
                data_post.append([latitude, longitude, month, day_week, direction] + s1.tolist() + s2.tolist() + s3.tolist() + [t])


    return np.array(data_post)

def data_loader(data_post, longueur_serie):
    """
    On construit nos DataLoaders de post que nous utiliserons
    pour itérer sur les données pour l'apprentissage de modèles.

    Parameters
    ----------
    data_post : TYPE array
        DESCRIPTION. Matrix of post data
    data_test : TYPE array
        DESCRIPTION. Matrix of test data
    longueur_serie : TYPE int
        DESCRIPTION. Length of series
    batch_size : TYPE int, optional
        DESCRIPTION. The default is 128.

    Returns
    -------
    data_loader_post : TYPE DataLoader
        DESCRIPTION. Post DataLoader

    """

    n_post = data_post.shape[0]

    latitude_post = data_post[:,0]
    longitude_post = data_post[:,1]
    
    month_post = np.zeros(((n_post, 12)))
    day_week_post = np.zeros(((n_post, 7)))
    direction_post = np.zeros(((n_post, 5)))
    
    # On crée les one-hot vectors pour les données post de mois, jour de la semaine, direction
    for index, elements in enumerate(data_post):
        month_post[index] = np.eye(12)[int(round(elements[2]))] # On utilise int(round(...)) à cause des erreurs d'arrondis parfois avec les float
        day_week_post[index] = np.eye(7)[int(round(elements[3]))]
        direction_post[index] = np.eye(5)[int(round(elements[4]))]
    
    serie_J_post = data_post[:,-3*longueur_serie:-1-2*longueur_serie]
    serie_J_moins_1_post = data_post[:,-1-2*longueur_serie:-1-longueur_serie]
    serie_J_moins_7_post = data_post[:,-1-longueur_serie:-1]
    target_post = data_post[:,-1]
    
    data_loader_post = torch.utils.data.DataLoader(list(zip(zip(latitude_post, longitude_post, month_post, day_week_post, direction_post, serie_J_post, serie_J_moins_1_post, serie_J_moins_7_post), target_post)), batch_size=n_post)

    
    return data_loader_post

def data_processing(data_loader_post, model):
    for (latitude, longitude, month, day_week, direction, serie_J, serie_J_moins_1, serie_J_moins_7), target in data_loader_post:

        latitude = latitude.float().unsqueeze(1)
        longitude = longitude.float().unsqueeze(1)
        month = month.float()
        day_week = day_week.float()
        direction = direction.float()
        serie_J = serie_J.float().unsqueeze(1)
        serie_J_moins_1 = serie_J_moins_1.float().unsqueeze(1)
        serie_J_moins_7 = serie_J_moins_7.float().unsqueeze(1)
        target = target.float().unsqueeze(1)

        return model.forward(latitude, longitude, month, day_week, direction, serie_J, serie_J_moins_1, serie_J_moins_7).view(-1)

def plot(data_post, output):
    plt.plot(data_post[:,-1])
    plt.plot(output.detach())
    plt.show()
