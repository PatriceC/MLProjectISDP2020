# -*- coding: utf-8 -*-
"""
Created on Thu Dec 9 23:30:00 2020

@author: Patrice CHANOL & Corentin MORVAN--CHAUMEIL
"""

import pandas as pd
import numpy as np

import torch

import data_preprocessing
import LSTM_seq_to_seq

def pred_vs_reality(model, input_window, output_window, date_range=['2018-07-09', '2018-08-10'], latitude=30.268652000000003, longitude=-97.759929, direction=0):
    """
    Affiche les prédictions du modèle choisi pour la date, lieu et direction choisis

    Parameters
    ----------
    model : Modèle choisi
    input_window : int
        Représente le nombre de jours de la séquence d'entrée qui servira d'input pour l'entraînement
    output_window : int
        Représente le nombre d'heures de la séquence de sortie qui servira de target
    date_range : list, optional
        DESCRIPTION. The default is ['2018-07-09', '2018-08-10']
    direction : int, optional
        DESCRIPTION. The default is 0
    latitude : float, optional
        DESCRIPTION. The default is 30.268652000000003
    longitude : float, optional
        DESCRIPTION. The default is -97.759929
    """

    data = pd.read_csv('./Radar_Traffic_Counts.csv')

    # Préparation du Dataset
    data = data.drop(columns=['location_name', 'Time Bin'])
    data['Direction'] = data['Direction'].astype('category').cat.codes
    data['Date'] = pd.to_datetime(data[['Year', 'Month', 'Day']], errors='coerce')
    data = data.sort_values(['Date'])

    # On ne garde que les données qui nous intéressent; c'est à dire celle de la range de date, lieu et direction
    data = data[(data['Date'] >= date_range[0]) & (data['Date'] <= date_range[1]) & (data['location_longitude'] == longitude) & (data['location_latitude'] == latitude) & (data['Direction'] == direction)]
    col = ['location_latitude', 'location_longitude', 'Year', 'Month', 'Day', 'Date', 'Day of Week', 'Hour', 'Direction']
    col_no_hour = ['location_latitude', 'location_longitude', 'Year', 'Month', 'Day', 'Date', 'Day of Week', 'Direction']
    data = data.groupby(col)['Volume'].sum().reset_index()
    # On va normaliser (méthode min-max) les valeurs
    volume_max, volume_min = data['Volume'].max(), data['Volume'].min()
    data = data.pivot_table(index=col_no_hour, columns='Hour', values='Volume').reset_index()
    print(data)

    data = data.dropna()

    for _, row in data.iterrows():
        # On récuprère les informations de cette donnée
        date, day_of_week = row['Date'], row['Day of Week']
        # On génère les séries
        result = data_preprocessing.series(date=date, latitude=latitude, longitude=longitude, direction=direction, input_window=input_window, output_window=output_window, data=data)

        if result is not None:
            target, serie = result
            target_norm, serie_norm = torch.tensor((target - volume_min)/(volume_max - volume_min)), torch.tensor((serie - volume_min)/(volume_max - volume_min))
            
            day_of_week_one_hot = np.zeros((serie_norm.shape[0], 7))
            for i in range(serie_norm.shape[0]):
                day_of_week_one_hot[i] = np.eye(7)[day_of_week]
            day_of_week_one_hot = torch.tensor(day_of_week_one_hot)
            
            pred_norm = model.forward(day_of_week_one_hot, serie_norm).detach()
            
            print(target_norm.tolist())
            print(pred_norm.tolist())
            return(0)

input_window = 7
output_window = 24
model = LSTM_seq_to_seq.LSTM(input_window, output_window)

pred_vs_reality(model, input_window, output_window)