# -*- coding: utf-8 -*-
"""
Created on Thu Dec 9 23:30:00 2020

@author: Patrice CHANOL & Corentin MORVAN--CHAUMEIL
"""

import pandas as pd
import numpy as np
import torch

import matplotlib.pyplot as plt

import data_preprocessing


def pred_vs_reality(model, input_window, output_window, date_range=['2018-07-09', '2018-08-10'], latitude=30.268652000000003, longitude=-97.759929, direction=0, epoch=None, pourcentage=None):
    """
    Affiche les prédictions du modèle choisi pour la date, lieu et direction choisis.

    Parameters
    ----------
    model : Modèle choisi
    input_window : int
        Représente le nombre de jours de la séquence d'entrée qui servira d'input pour l'entraînement
    output_window : int
        Représente le nombre d'heures de la séquence de sortie qui servira de target
    date_range : list, optional
        DESCRIPTION. The default is ['2018-07-09', '2018-08-10']
    latitude : float, optional
        DESCRIPTION. The default is 30.268652000000003
    longitude : float, optional
        DESCRIPTION. The default is -97.759929
    direction : int, optional
        DESCRIPTION. The default is 0
    epoch : int
        Nombre d'epoch dans l'apprentissage du modèle
    pourcentage : int
        Pourcentage de l'epoch d'apprentissage
    """
    data = pd.read_csv('./Radar_Traffic_Counts.csv')

    # Préparation du Dataset
    data = data.drop(columns=['location_name', 'Time Bin'])
    data['Direction'] = data['Direction'].astype('category').cat.codes
    data['Date'] = pd.to_datetime(data[['Year', 'Month', 'Day']], errors='coerce')
    data = data.sort_values(['Date'])
    data_tot = data

    # Le dataprocessing se passe de manière très analogue à data_preprocessing.py mais on ne peut pas le reprendre car il y a tout de même quelques différences
    # On ne garde que les données qui nous intéressent; c'est à dire celle de la range de date, lieu et direction
    data = data[(data['Date'] >= date_range[0]) & (data['Date'] <= date_range[1]) & (data['location_longitude'] == longitude) & (data['location_latitude'] == latitude) & (data['Direction'] == direction)]
    col = ['location_latitude', 'location_longitude', 'Year', 'Month', 'Day', 'Date', 'Day of Week', 'Hour', 'Direction']
    col_no_hour = ['location_latitude', 'location_longitude', 'Year', 'Month', 'Day', 'Date', 'Day of Week', 'Direction']
    data = data.groupby(col)['Volume'].sum().reset_index()
    data_tot = data_tot.groupby(col)['Volume'].sum().reset_index()  # On garde les données totales car on aura besoin d'avoir accès aux jours avant les jours auxquels on souhaite faire des prédictions
    # On va normaliser (méthode min-max) les valeurs
    volume_max, volume_min = data['Volume'].max(), data['Volume'].min()
    data = data.pivot_table(index=col_no_hour, columns='Hour', values='Volume').reset_index()
    data_tot = data_tot.pivot_table(index=col_no_hour, columns='Hour', values='Volume').reset_index()

    # De même, on supprime les lignes avec des valeurs manquantes
    data = data.dropna()
    data_tot = data_tot.dropna()

    # Ce sont les listes qui contiendront les dates et heures (x), volumes réels (t) et volume prédits (p)
    x, t, p = [], [], []

    # Tous les intervalles de temps output_window, on va réaliser des prédictions seq-to-seq sur les output_window heures suivantes, et enregistrer ces prédictions pour les comparer, visuellement, aux valeurs réelles
    current_date = pd.to_datetime(date_range[0])
    while current_date < pd.to_datetime(date_range[1]):

        date_row = current_date.replace(hour=0)
        # On enregistre l'heure actuelle
        hour_current = current_date.hour

        # On regarde sur quelle ligne on se trouve
        row = data[(data['Date'] == date_row)]
        day_of_week = row['Day of Week']

        # On va créer nos séries de données pour le jour désiré et l'output_window
        result = data_preprocessing.series(date=date_row, latitude=latitude, longitude=longitude, direction=direction, input_window=input_window, output_window=output_window, data=data_tot)

        if result is not None:
            target, serie = result
            # On récupère les séries de valeurs à partir de l'heure qui nous intéresse, c'est-à-dire hour_current, puis on normalise pour inférer sur le modèle
            target_norm, serie_norm = torch.tensor((target[hour_current] - volume_min)/(volume_max - volume_min)), torch.tensor((serie[hour_current] - volume_min)/(volume_max - volume_min)).unsqueeze(0)

            # On transforme notre jour de la semaine en matrice de one-hot vectors (toutes les lignes sont donc les mêmes)
            day_of_week_one_hot = torch.tensor(np.eye(7)[day_of_week])

            # On infère pour obtenir nos prédictions
            pred_norm = model.forward(day_of_week_one_hot, serie_norm.float()).detach().squeeze(0)

            # On dénormalise
            target_current = (target_norm * (volume_max - volume_min) + volume_min).tolist()
            pred_current = (pred_norm * (volume_max - volume_min) + volume_min).tolist()

            x += [current_date + pd.to_timedelta(h, unit='h') for h in range(output_window)]
            t += target_current
            p += pred_current

        # On incrémente la date de l'intervalle output_window
        current_date += pd.to_timedelta(output_window, unit='h')

    # On visualise les prédictions vs la réalité et on enregistre le graphe

    res = pd.DataFrame()
    res['x'] = x
    res['t'] = t
    res['p'] = p
    res.index = res['x']
    plt.figure(0)
    res['t'].plot(color="green", label='Donnée réelle')
    res['p'].plot(color="red", label='Prediction')
    plt.title(model.name_model +': Data vs Pred')
    plt.axis([x[0], x[-1], 0, max(max(t), max(p))])
    plt.legend(loc='upper right')
    if epoch is None:
        plt.show()
    else:
        plt.show()
        plt.savefig('./visu/pred_vs_data_' + model.name_model + '_epoch_' + str(epoch) + '_pourcentage_' + str(pourcentage) + '%_' + str(input_window) + '_days_to_' + str(output_window) + '_hours.png', dpi=300)
        plt.close()


def forecast(model, input_window, output_window, date_range=['2018-07-09', '2018-08-10'], latitude=30.268652000000003, longitude=-97.759929, direction=0, epoch=None):
    """
    A partir de la date date_range[0], réalise un forecast avec le modèle choisi jusqu'à la date date_range[1].

    Pour cela, on ne prend que les données nécessaires pour la première prédiction (prédiction des premières 24h par exemple)
    puis utilise les prédictions pour réaliser les prédictions postérieures (pendant plusieurs semaines par exemple)
    Les prédictions sont donc biaisées au fur et à mesure; et l'on va tenter de représenter visuellement ici à quel
    point le forecast long-terme est proche de la réalité ou non.

    Parameters
    ----------
    model : Modèle choisi
    input_window : int
        Représente le nombre de jours de la séquence d'entrée qui servira d'input pour l'entraînement
    output_window : int
        Représente le nombre d'heures de la séquence de sortie qui servira de target
    date_range : list, optional
        DESCRIPTION. The default is ['2018-07-09', '2018-08-10']
    latitude : float, optional
        DESCRIPTION. The default is 30.268652000000003
    longitude : float, optional
        DESCRIPTION. The default is -97.759929
    direction : int, optional
        DESCRIPTION. The default is 0
    epoch : int
        Nombre d'epoch dans l'apprentissage du modèle
    """
    data = pd.read_csv('./Radar_Traffic_Counts.csv')

    # Préparation du Dataset
    data = data.drop(columns=['location_name', 'Time Bin'])
    data['Direction'] = data['Direction'].astype('category').cat.codes
    data['Date'] = pd.to_datetime(data[['Year', 'Month', 'Day']], errors='coerce')
    data = data.sort_values(['Date'])
    data_tot = data

    # Le dataprocessing se passe de manière très analogue à data_preprocessing.py mais on ne peut pas le reprendre car il y a tout de même quelques différences
    # On ne garde que les données qui nous intéressent; c'est à dire celle de la range de date, lieu et direction
    data = data[(data['Date'] >= date_range[0]) & (data['Date'] <= date_range[1]) & (data['location_longitude'] == longitude) & (data['location_latitude'] == latitude) & (data['Direction'] == direction)]
    col = ['location_latitude', 'location_longitude', 'Year', 'Month', 'Day', 'Date', 'Day of Week', 'Hour', 'Direction']
    col_no_hour = ['location_latitude', 'location_longitude', 'Year', 'Month', 'Day', 'Date', 'Day of Week', 'Direction']
    data = data.groupby(col)['Volume'].sum().reset_index()
    data_tot = data_tot.groupby(col)['Volume'].sum().reset_index()  # On garde les données totales car on aura besoin d'avoir accès aux jours avant les jours auxquels on souhaite faire des prédictions
    # On va normaliser (méthode min-max) les valeurs
    volume_max, volume_min = data['Volume'].max(), data['Volume'].min()
    data = data.pivot_table(index=col_no_hour, columns='Hour', values='Volume').reset_index()
    data_tot = data_tot.pivot_table(index=col_no_hour, columns='Hour', values='Volume').reset_index()

    # De même, on supprime les lignes avec des valeurs manquantes
    data = data.dropna()
    data_tot = data_tot.dropna()

    # Ce sont les listes qui contiendront les dates et heures (x), volumes réels (t) et volume prédits (p)
    x, t, p = [], [], []

    current_date = pd.to_datetime(date_range[0])
    inputs_for_prediction, output_of_prediction = None, None
    while current_date < pd.to_datetime(date_range[1]):

        date_row = current_date.replace(hour=0)
        # On enregistre l'heure actuelle
        hour_current = current_date.hour

        # On regarde sur quelle ligne on se trouve
        row = data[(data['Date'] == date_row)]
        day_of_week = row['Day of Week']

        # On va créer nos séries de données pour le jour désiré et l'output_window
        result = data_preprocessing.series(date=date_row, latitude=latitude, longitude=longitude, direction=direction, input_window=input_window, output_window=output_window, data=data_tot)

        if result is not None:
            # On ne servira des données de "serie" qu'au premier passage; après nous aurons des données d'entrées correspondant au décalage que nous avons effectué en prenant en input les valeurs prédites précédemment
            target, serie = result

            # Au premier passage
            if inputs_for_prediction is None:
                target, serie = target[hour_current], serie[hour_current]
                inputs_for_prediction = serie
            # Ensuite, on modifie la série d'inputs pour ajouter petit à petit les prédictions faites dans les itérations précédentes
            else:
                target = target[hour_current]
                inputs_for_prediction = np.array(list(inputs_for_prediction[output_window:]) + output_of_prediction)

            # On récupère les séries de valeurs à partir de l'heure qui nous intéresse, c'est-à-dire hour_current, puis on normalise pour inférer sur le modèle
            target_norm, input_norm = torch.tensor((target - volume_min)/(volume_max - volume_min)), torch.tensor((inputs_for_prediction - volume_min)/(volume_max - volume_min)).unsqueeze(0)

            # On transforme notre jour de la semaine en matrice de one-hot vectors (toutes les lignes sont donc les mêmes)
            day_of_week_one_hot = torch.tensor(np.eye(7)[day_of_week])

            # On infère pour obtenir nos prédictions
            pred_norm = model.forward(day_of_week_one_hot, input_norm.float()).detach().squeeze(0)

            # On dénormalise
            target_current = (target_norm * (volume_max - volume_min) + volume_min).tolist()
            pred_current = (pred_norm * (volume_max - volume_min) + volume_min).tolist()

            # Liste de prédictions qui seront nécessaires en input pour la prochaine prédiction du forecast
            output_of_prediction = pred_current

            x += [current_date + pd.to_timedelta(h, unit='h') for h in range(output_window)]
            t += target_current
            p += pred_current

        # On incrémente la date de l'intervalle output_window
        current_date += pd.to_timedelta(output_window, unit='h')

    # On visualise les prédictions vs la réalité et on enregistre le graphe
    res = pd.DataFrame()
    res['x'] = x
    res['t'] = t
    res['p'] = p
    res.index = res['x']
    plt.figure(0)
    res['t'].plot(color="green", label='Donnée réelle')
    res['p'].plot(color="red", label='Prediction')
    plt.title(model.name_model +': Data vs Pred')
    plt.axis([x[0], x[-1], 0, max(max(t), max(p))])
    plt.legend(loc='upper right')
    if epoch is None:
        plt.show()
    else:
        plt.show()
        plt.savefig('./visu/forecast_' + model.name_model + '_epoch_' + str(epoch) + '_' + str(input_window) + '_days_to_' + str(output_window) + '_hours.png', dpi=300)
        plt.close()
