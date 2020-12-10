# -*- coding: utf-8 -*-
"""
Created on Fri Nov 27 19:01:58 2020

@author: Patrice CHANOL & Corentin MORVAN--CHAUMEIL
"""

import pandas as pd
import numpy as np

import torch

import random

def series(date, latitude, longitude, direction, input_window, output_window, data):
    """
    Parameters
    ----------
    date : TYPE datetime
        DESCRIPTION. Date des séries à prédire
    latitude : TYPE float
        DESCRIPTION. Latitude des séries
    longitude : TYPE float
        DESCRIPTION. Longitude des séries
    direction : TYPE int
        DESCRIPTION. Direction des séries
    input_window : TYPE int
        DESCRIPTION. Nombre de jours de la série d'input de la prédiction
    output_window: TYPE int
        DESCRIPTION. Nombre d'heures de la série d'output de la prédiction
    data : TYPE DataFrame
        DESCRIPTION. Donnée
    """
    # Récupération des données aux jours nécessaires pour le lieu et direction souhaités
    dict_J = {}
    for j in range(input_window, 0, -1):
        dict_J['row_J_moins_' + str(j)] = data[(data['location_latitude'] == latitude) & (data['location_longitude'] == longitude) & (data['Date'] == date - pd.to_timedelta(j, unit='d')) & (data['Direction'] == direction)]
    dict_J['row_J'] = data[(data['location_latitude'] == latitude) & (data['location_longitude'] == longitude) & (data['Date'] == date) & (data['Direction'] == direction)]
    for j in range(1, 2 + (output_window - 2) // 24):
        dict_J['row_J_plus_' + str(j)] = data[(data['location_latitude'] == latitude) & (data['location_longitude'] == longitude) & (data['Date'] == date + pd.to_timedelta(j, unit='d')) & (data['Direction'] == direction)]

    # Si on a pas de donnée pour un des jours
    for element in dict_J:
        if dict_J[element].empty:
            return(None)

    # Sinon on dispose de toutes les données et on peut générer les séries
    serie_totale = []
    for element in dict_J:
        serie_totale += dict_J[element].values.tolist()[0][8:]

    nb_series = 24
    target, serie = np.zeros((nb_series, output_window)), np.zeros((nb_series, input_window * nb_series))

    for h in range(nb_series):
        target[h] = serie_totale[h + input_window * nb_series : h + input_window * nb_series + output_window]
        serie[h] = serie_totale[h : h + input_window * nb_series]
        
    return(target, serie)


def process_data(input_window=7, output_window=24, file='./Radar_Traffic_Counts.csv'):
    """
    Génération du Dataset désiré.

    Parameters
    ----------
    input_window : int, optional, default=7
        Représente le nombre de jours de la séquence d'entrée qui servira d'input pour l'entraînement.
    output_window : int, optional, default=24
        Représente le nombre d'heures de la séquence de sortie qui servira de target.
    file : String, optional, default='./Radar_Traffic_Counts.csv'
        Fichier de données.

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
    # On normalise les données avec la méthode min-max
    volume_max, volume_min = data['Volume'].max(), data['Volume'].min()
    data = data.pivot_table(index=col_no_hour, columns='Hour', values='Volume').reset_index()
    # Suppression des jours contenant des données manquantes
    data = data.dropna()

    data_train, data_test = [], []
    # Pour chaque donnée: jour précis, lieu précis, direction précise
    for _, row in data.iterrows():
        # On récuprère les informations de cette donnée
        latitude, longitude = row['location_latitude'], row['location_longitude']
        date, day_of_week = row['Date'], row['Day of Week']
        direction = row['Direction']
        # On génère les séries
        result = series(date=date, latitude=latitude, longitude=longitude, direction=direction, input_window=input_window, output_window=output_window, data=data)

        if result is not None:
            day_of_week_one_hot = list(np.eye(7)[day_of_week])
            target, serie = result
            for t, s in zip(target, serie):
                # On normalise les valeurs
                s_norm = list((s - volume_min)/(volume_max - volume_min))
                t_norm = list((t - volume_min)/(volume_max - volume_min))
                # On sépare le dataset en 90% training et 10% test
                if random.random() < 0.9:
                    data_train.append(day_of_week_one_hot + s_norm + t_norm)
                else:
                    data_test.append(day_of_week_one_hot + s_norm + t_norm)
    # Mélange des données
    random.shuffle(data_train)
    random.shuffle(data_test)

    data_train = np.array(data_train)
    data_test = np.array(data_test)
    # Enregistrement des données
    torch.save(data_train, './data/data_train_' + str(input_window) + '_days_to_' + str(output_window) + '_hours.txt')
    torch.save(data_test, './data/data_test_' + str(input_window) + '_days_to_' + str(output_window) + '_hours.txt')
    
    return(data_train, data_test)

def data_loader(data_train, data_test, input_window, output_window, batch_size=128):
    """
    On construit nos DataLoaders de train/test que nous utiliserons
    pour itérer sur les données pour l'apprentissage de modèles.

    Parameters
    ----------
    data_train : TYPE array
        DESCRIPTION. Matrix of train data
    data_test : TYPE array
        DESCRIPTION. MAtrix of test data
    input_window : TYPE int
        DESCRIPTION. Nombre de jours de la série d'input de la prédiction
    output_window: TYPE int
        DESCRIPTION. Nombre d'heures de la série d'output de la prédiction
    batch_size : TYPE int, optional
        DESCRIPTION. The default is 128.

    Returns
    -------
    data_loader_train : TYPE DataLoader
        DESCRIPTION. Train DataLoader
    data_loader_test : TYPE DataLoader
        DESCRIPTION. Test DataLoader
    """
    day_of_week_train = data_train[:,:7]
    day_of_week_test = data_test[:,:7]
    serie_train = data_train[:,7:-output_window]
    serie_test = data_test[:,7:-output_window]
    target_train = data_train[:,-output_window:]
    target_test = data_test[:,-output_window:]

    data_loader_train = torch.utils.data.DataLoader(list(zip(zip(day_of_week_train, serie_train), target_train)), batch_size=batch_size, shuffle=True)
    data_loader_test = torch.utils.data.DataLoader(list(zip(zip(day_of_week_test, serie_test), target_test)), batch_size=batch_size, shuffle=True)

    return(data_loader_train, data_loader_test)