# -*- coding: utf-8 -*-
"""
Created on Fri Nov 27 19:01:58 2020

@author: Patrice CHANOL & Corentin MORVAN--CHAUMEIL
"""

import pandas as pd
import numpy as np

import torch

import random


def series(Date_J, latitude, longitude, direction, nb_days_before, data):
    """
    Retourne 3 séries de longueur_serie valeurs de Volume pour un jour,
    une position, et une direction

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
    """
    # Récupération des données de Date_J (au jour J, J-1, J-2, J-7, J-8)
    dict_J = {}
    for j in range(nb_days_before, 0, -1):
        dict_J['row_J_moins_' + str(j)] = data[(data['location_latitude'] == latitude) & (data['location_longitude'] == longitude) & (data['Date'] == Date_J - pd.to_timedelta(j, unit='d')) & (data['Direction'] == direction)]
    dict_J['row_J'] = data[(data['location_latitude'] == latitude) & (data['location_longitude'] == longitude) & (data['Date'] == Date_J) & (data['Direction'] == direction)]
    dict_J['row_J_plus_1'] = data[(data['location_latitude'] == latitude) & (data['location_longitude'] == longitude) & (data['Date'] == Date_J + pd.to_timedelta(1, unit='d')) & (data['Direction'] == direction)]

    # Si on a pas de donnée pour un des jours
    for element in dict_J:
        if dict_J[element].empty:
            return(None)

    # Sinon on dispose de toutes les données et on peut générer des séries
    serie_totale = []
    for element in dict_J:
        serie_totale += dict_J[element].values.tolist()[0][8:]

    nb_series, nb_hour = 24, 24
    target, serie = np.zeros((nb_series, nb_hour)), np.zeros((nb_series, nb_days_before * nb_hour))

    for h in range(nb_series):
        target[h] = serie_totale[h + nb_days_before * nb_hour : h + nb_days_before * nb_hour + 24]
        serie[h] = serie_totale[h : h + nb_days_before * nb_hour]
        
    return(target, serie)


def process_data(nb_days_before=7, file='./Radar_Traffic_Counts.csv'):
    """
    Génération du Dataset désiré.

    Parameters
    ----------
    nb_days_before : TYPE, optional
        DESCRIPTION. The default is 7.
    file : TYPE, optional
        DESCRIPTION. The default is './Radar_Traffic_Counts.csv'.

    """
    data = pd.read_csv(file)

    # Préparation du Dataset
    data = data.drop(columns=['location_name', 'Time Bin'])
    data['Direction'] = data['Direction'].astype('category').cat.codes
    data['Date'] = pd.to_datetime(data[['Year', 'Month', 'Day']], errors='coerce')
    data['location_latitude'] = data['location_latitude']
    data['location_longitude'] = data['location_longitude']
    col = ['location_latitude', 'location_longitude', 'Year', 'Month', 'Day', 'Date', 'Day of Week', 'Hour', 'Direction']
    col_no_hour = ['location_latitude', 'location_longitude', 'Year', 'Month', 'Day', 'Date', 'Day of Week', 'Direction']
    data = data.groupby(col)['Volume'].sum().reset_index()
    # On va normaliser (méthode min-max) les valeurs
    volume_max, volume_min = data['Volume'].max(), data['Volume'].min()
    data = data.pivot_table(index=col_no_hour, columns='Hour', values='Volume').reset_index()
    # Suppression des jours contenant des données manquantes
    data = data.dropna()

    data_train, data_test = [], []
    # Pour chaque jour
    for _, row in data.iterrows():
        # On récuprère les informations de cette donnée
        latitude, longitude = row['location_latitude'], row['location_longitude']
        date = row['Date']
        direction = row['Direction']
        # On génère les séries
        result = series(Date_J=date, latitude=latitude, longitude=longitude, direction=direction, nb_days_before=nb_days_before, data=data)

        if result is not None:
            # On normalise (méthode min-max) les valeurs de latitude et longitude
            target, serie = result
            # On récupère les heures pour plot
            for t, s in zip(target, serie):
                # On normalise les valeurs
                s_norm = list((s - volume_min)/(volume_max - volume_min))
                t_norm = list((t - volume_min)/(volume_max - volume_min))
                # On sépare le dataset en 90% training et 10% test
                if random.random() < 0.9:
                    data_train.append(s_norm + t_norm)
                else:
                    data_test.append(s_norm + t_norm)
    # Mélange des données
    random.shuffle(data_train)
    random.shuffle(data_test)

    data_train = np.array(data_train)
    data_test = np.array(data_test)
    # Enregistrement des données
    torch.save(data_train, 'data_seq_train_' + str(nb_days_before) + '.txt')
    torch.save(data_test, 'data_seq_test_' + str(nb_days_before) + '.txt')

    return(data_train, data_test)


def data_loader(data_train, data_test, batch_size=128):
    """
    On construit nos DataLoaders de train/test que nous utiliserons
    pour itérer sur les données pour l'apprentissage de modèles.

    Parameters
    ----------
    data_train : TYPE array
        DESCRIPTION. Matrix of train data
    data_test : TYPE array
        DESCRIPTION. MAtrix of test data
    batch_size : TYPE int, optional
        DESCRIPTION. The default is 128.

    Returns
    -------
    data_loader_train : TYPE DataLoader
        DESCRIPTION. Train DataLoader
    data_loader_test : TYPE DataLoader
        DESCRIPTION. Test DataLoader

    """
    serie_train = data_train[:, :-24]
    serie_test = data_test[:, :-24]

    target_train = data_train[:, -24:]
    target_test = data_test[:, -24:]

    data_loader_train = torch.utils.data.DataLoader(list(zip(serie_train, target_train)), batch_size=batch_size, shuffle=True)
    data_loader_test = torch.utils.data.DataLoader(list(zip(serie_test, target_test)), batch_size=batch_size, shuffle=True)

    return data_loader_train, data_loader_test
