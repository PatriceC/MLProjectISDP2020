# -*- coding: utf-8 -*-
"""
Created on Fri Nov 27 19:01:58 2020

@author: Patrice CHANOL & Corentin MORVAN--CHAUMEIL
"""

import pandas as pd
import numpy as np

import torch

import random

def series(Date_J, latitude, longitude, direction, longueur_serie, data):
    '''
    Retourne 3 séries de longueur_serie valeurs de Volume pour un jour, une position, et une direction

    Parameters
    ----------
    Date_J : TYPE datetime
        DESCRIPTION. Date des séries à prédire
    latitude : TYPE float
        DESCRIPTION. Latitude des séries
    longitude : TYPE float
        DESCRIPTION. Longitude des séries
    direction : TYPE int
        DESCRIPTION. Direction des séries
    longueur_serie : TYPE int
        DESCRIPTION. Longueur des séries
    data : TYPE DataFrame
        DESCRIPTION. Donnée
    '''

    # Récupération des données de Date_J (au jour J, J-1, J-2, J-7, J-8)
    row_J = data[(data['location_latitude'] == latitude) & (data['location_longitude'] == longitude) & (data['Date'] == Date_J) & (data['Direction'] == direction)]
    row_J_moins_1 = data[(data['location_latitude'] == latitude) & (data['location_longitude'] == longitude) & (data['Date'] == Date_J - pd.to_timedelta(1, unit='d')) & (data['Direction'] == direction)]
    row_J_moins_2 = data[(data['location_latitude'] == latitude) & (data['location_longitude'] == longitude) & (data['Date'] == Date_J - pd.to_timedelta(2, unit='d')) & (data['Direction'] == direction)]
    row_J_moins_7 = data[(data['location_latitude'] == latitude) & (data['location_longitude'] == longitude) & (data['Date'] == Date_J - pd.to_timedelta(7, unit='d')) & (data['Direction'] == direction)]
    row_J_moins_8 = data[(data['location_latitude'] == latitude) & (data['location_longitude'] == longitude) & (data['Date'] == Date_J - pd.to_timedelta(8, unit='d')) & (data['Direction'] == direction)]

    # Si on a pas de donnée pour les jour J, J-1, J-7, on ne renvoit rien
    if (row_J.empty or row_J_moins_1.empty or row_J_moins_7.empty):
        return None

    # Sinon si on a pas de données pour les jour J-2, J-8, on renvoit juste les séries disponibles dans la journée 
    elif (row_J_moins_2.empty or row_J_moins_8.empty):
        valeurs_J, valeurs_J_moins_1, valeurs_J_moins_7 = row_J.values.tolist()[0][8:], row_J_moins_1.values.tolist()[0][8:], row_J_moins_7.values.tolist()[0][8:]

        nb_series = 24 - longueur_serie + 1
        target, serie_J, serie_J_moins_1, serie_J_moins_7 = np.zeros(nb_series), np.zeros((nb_series,longueur_serie-1)), np.zeros((nb_series,longueur_serie)), np.zeros((nb_series,longueur_serie))

        for h in range(nb_series):
            target[h] = valeurs_J[h + longueur_serie - 1]
            serie_J[h] = valeurs_J[h : h + longueur_serie - 1]
            serie_J_moins_1[h] = valeurs_J_moins_1[h : h + longueur_serie]
            serie_J_moins_7[h] = valeurs_J_moins_7[h : h + longueur_serie]

    # Sinon on dispose de toute les données et on peut générer 24 séries
    else:
        valeurs_J, valeurs_J_moins_1, valeurs_J_moins_2, valeurs_J_moins_7, valeurs_J_moins_8 = row_J.values.tolist()[0][8:], row_J_moins_1.values.tolist()[0][8:], row_J_moins_2.values.tolist()[0][8:], row_J_moins_7.values.tolist()[0][8:], row_J_moins_8.values.tolist()[0][8:]
        V_J, V_J_1, V_J_7 = valeurs_J_moins_1 + valeurs_J,  valeurs_J_moins_2 + valeurs_J_moins_1, valeurs_J_moins_8 + valeurs_J_moins_7

        nb_series = 24
        target, serie_J, serie_J_moins_1, serie_J_moins_7 = np.zeros(nb_series), np.zeros((nb_series,longueur_serie-1)), np.zeros((nb_series,longueur_serie)), np.zeros((nb_series,longueur_serie))

        for h in range(nb_series):
            target[h] = V_J[h + nb_series]
            serie_J[h] = V_J[h + nb_series - longueur_serie + 1 : h + nb_series]
            serie_J_moins_1[h] = V_J_1[h + nb_series - longueur_serie : h + nb_series]
            serie_J_moins_7[h] = V_J_7[h + nb_series - longueur_serie : h + nb_series]
    
    return(target, serie_J, serie_J_moins_1, serie_J_moins_7)

def process_data(longueur_serie=24, file='./Radar_Traffic_Counts.csv'):
    """
    Génération du Dataset désiré 

    Parameters
    ----------
    date_range : TYPE, optional
        DESCRIPTION. The default is ['2017','2020'].
    direction : TYPE, optional
        DESCRIPTION. The default is None.
    latitude : TYPE, optional
        DESCRIPTION. The default is [-100,100].
    longitude : TYPE, optional
        DESCRIPTION. The default is [-100,100].
    longueur_serie : TYPE, optional
        DESCRIPTION. The default is 6.
    file : TYPE, optional
        DESCRIPTION. The default is './Radar_Traffic_Counts.csv'.

    """
    data = pd.read_csv(file)

    # On va normaliser (méthode min-max) les valeurs de latitude et longitude
    latitude_max, latitude_min = data['location_latitude'].max(), data['location_latitude'].min()
    longitude_max, longitude_min = data['location_longitude'].max(), data['location_longitude'].min()

    # Préparation du Dataset
    data = data.drop(columns=['location_name', 'Time Bin'])
    data['Direction'] = data['Direction'].astype('category').cat.codes
    data['Date'] = pd.to_datetime(data[['Year', 'Month', 'Day']], errors = 'coerce')
    data['location_latitude'] = data['location_latitude'] * (10**14)
    data['location_longitude'] = data['location_longitude'] * (10**14)
    col = ['location_latitude', 'location_longitude', 'Year', 'Month', 'Day', 'Date', 'Day of Week', 'Hour', 'Direction']
    col_no_hour = ['location_latitude', 'location_longitude', 'Year', 'Month', 'Day', 'Date', 'Day of Week', 'Direction']
    data = data.groupby(col)['Volume'].sum().reset_index()
    # On va normaliser (méthode min-max) les valeurs
    volume_max, volume_min = data['Volume'].max(), data['Volume'].min()
    data = data.pivot_table(index=col_no_hour, columns='Hour', values='Volume').reset_index()
    # Suppression des jours contenant des données manquantes
    data = data.dropna()
    # On garde les valeurs de mois entre 0 et 11 (plutôt que 1 et 12), ce qui sera plus pratique pour créer des one-hot vectors
    data['Month'] = data['Month'] - 1

    data_train, data_test = [], []
    # Pour chaque jour
    for _, row in data.iterrows():
        # On récuprère les informations de cette donnée
        latitude, longitude = row['location_latitude'], row['location_longitude']
        month, day_week, date = row['Month'], row['Day of Week'], row['Date']
        direction = row['Direction']
        # On génère les séries
        result = series(Date_J=date, latitude=latitude, longitude=longitude, direction=direction, longueur_serie=longueur_serie, data=data)
    
        # On normalise (méthode min-max) les valeurs de latitude et longitude
        latitude = (latitude - latitude_min)/(latitude_max - latitude_min)
        longitude = (longitude - longitude_min)/(longitude_max - longitude_min)
        if result is not None:
            target, serie_J, serie_J_moins_1, serie_J_moins_7 = result
            # On récupère les heures pour plot
            for t, s1, s2, s3 in zip(target, serie_J, serie_J_moins_1, serie_J_moins_7):
                # On normalise les valeurs
                s1_norm = list((s1 - volume_min)/(volume_max - volume_min))
                s2_norm = list((s2 - volume_min)/(volume_max - volume_min))
                s3_norm = list((s3 - volume_min)/(volume_max - volume_min))
                t_norm = (t - volume_min)/(volume_max - volume_min)
                # On sépare le dataset en 90% training et 10% test
                if random.random() < 0.9:
                    data_train.append([latitude, longitude, month, day_week, direction] + s1_norm + s2_norm + s3_norm + [t_norm])
                else:
                    data_test.append([latitude, longitude, month, day_week, direction] + s1_norm + s2_norm + s3_norm + [t_norm])
    # Mélange des données
    random.shuffle(data_train)
    random.shuffle(data_test)
    data_train = torch.tensor(data_train)
    data_test = torch.tensor(data_test)
    # Enregistrement des données
    torch.save(data_train, 'data_train_' + str(longueur_serie) + '.txt')
    torch.save(data_test, 'data_test_' + str(longueur_serie) + '.txt')

    return(data_train, data_test)

def data_loader(data_train, data_test, longueur_serie, batch_size = 128):
    """
    On construit nos DataLoaders de train/test que nous utiliserons
    pour itérer sur les données pour l'apprentissage de modèles.

    Parameters
    ----------
    data_train : TYPE array
        DESCRIPTION. Matrix of train data
    data_test : TYPE array
        DESCRIPTION. MAtrix of test data
    longueur_serie : TYPE int
        DESCRIPTION. Length of series
    batch_size : TYPE int, optional
        DESCRIPTION. The default is 128.

    Returns
    -------
    data_loader_train : TYPE DataLoader
        DESCRIPTION. Train DataLoader
    data_loader_test : TYPE DataLoader
        DESCRIPTION. Test DataLoader

    """

    n_train, n_test = data_train.shape[0], data_test.shape[0]

    latitude_train = data_train[:,0]
    latitude_test = data_test[:,0]
    longitude_train = data_train[:,1]
    longitude_test = data_test[:,1]
    
    month_train = np.zeros(((n_train, 12)))
    month_test = np.zeros((n_test, 12))
    day_week_train = np.zeros(((n_train, 7)))
    day_week_test = np.zeros(((n_train, 7)))
    direction_train = np.zeros(((n_train, 5)))
    direction_test = np.zeros(((n_train, 5)))
    
    # On crée les one-hot vectors pour les données train/test de mois, jour de la semaine, direction
    for index, elements in enumerate(data_train):
        month_train[index] = np.eye(12)[int(round(elements[2]))] # On utilise int(round(...)) à cause des erreurs d'arrondis parfois avec les float
        day_week_train[index] = np.eye(7)[int(round(elements[3]))]
        direction_train[index] = np.eye(5)[int(round(elements[4]))]
    for index, elements in enumerate(data_test):
        month_test[index] = np.eye(12)[int(round(elements[2]))]
        day_week_test[index] = np.eye(7)[int(round(elements[3]))]
        direction_test[index] = np.eye(5)[int(round(elements[4]))]
    
    serie_J_train = data_train[:,-3*longueur_serie:-1-2*longueur_serie]
    serie_J_test = data_test[:,-3*longueur_serie:-1-2*longueur_serie]
    serie_J_moins_1_train = data_train[:,-1-2*longueur_serie:-1-longueur_serie]
    serie_J_moins_1_test = data_test[:,-1-2*longueur_serie:-1-longueur_serie]
    serie_J_moins_7_train = data_train[:,-1-longueur_serie:-1]
    serie_J_moins_7_test = data_test[:,-1-longueur_serie:-1]
    target_train = data_train[:,-1]
    target_test = data_test[:,-1]
    
    data_loader_train = torch.utils.data.DataLoader(list(zip(zip(latitude_train, longitude_train, month_train, day_week_train, direction_train, serie_J_train, serie_J_moins_1_train, serie_J_moins_7_train), target_train)), batch_size= batch_size, shuffle=True)
    data_loader_test = torch.utils.data.DataLoader(list(zip(zip(latitude_test, longitude_test, month_test, day_week_test, direction_test, serie_J_test, serie_J_moins_1_test, serie_J_moins_7_test), target_test)), batch_size= batch_size, shuffle=True)

    return data_loader_train, data_loader_test
