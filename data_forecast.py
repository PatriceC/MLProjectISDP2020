# -*- coding: utf-8 -*-
"""
Created on Wed Dec  2 11:46:52 2020

@author: Patrice CHANOL & Corentin MORVAN--CHAUMEIL
"""

import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt

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

def forecast(model, date_range=['2018-07-09','2018-08-10'], direction=0, latitude=30.268652000000003, longitude=-97.759929, longueur_serie=24, file='./Radar_Traffic_Counts.csv'):
    '''
    Forecast des données à partir du premier élément de date_range jusqu'au dernier'

    Parameters
    ----------
    model : TYPE
        DESCRIPTION. Model utiliser pour Forecast
    date_range : TYPE, optional
        DESCRIPTION. The default is ['2018-07-09','2018-08-10'].
    direction : TYPE, optional
        DESCRIPTION. The default is 0.
    latitude : TYPE, optional
        DESCRIPTION. The default is 30.268652000000003.
    longitude : TYPE, optional
        DESCRIPTION. The default is -97.759929.
    longueur_serie : TYPE, optional
        DESCRIPTION. The default is 24.
    file : TYPE, optional
        DESCRIPTION. The default is './Radar_Traffic_Counts.csv'.

    '''

    data = pd.read_csv(file)

    # On va normaliser (méthode min-max) les valeurs de latitude et longitude
    latitude_max, latitude_min = data['location_latitude'].max(), data['location_latitude'].min()
    longitude_max, longitude_min = data['location_longitude'].max(), data['location_longitude'].min()

    # Préparation du Dataset
    data = data.drop(columns=['location_name', 'Time Bin'])
    data['Direction'] = data['Direction'].astype('category').cat.codes
    data['Date'] = pd.to_datetime(data[['Year', 'Month', 'Day']], errors = 'coerce')
    data = data.sort_values(['Date'])
    data_pred = data[(data['Date'] >= date_range[0]) & (data['Date'] <= date_range[1])]
    data_pred = data_pred[data_pred['location_latitude'] == latitude]
    data_pred = data_pred[data_pred['location_longitude'] == longitude]
    data_pred = data_pred[data_pred['Direction'] == direction]
    col = ['location_latitude', 'location_longitude', 'Year', 'Month', 'Day', 'Date', 'Day of Week', 'Hour', 'Direction']
    col_no_hour = ['location_latitude', 'location_longitude', 'Year', 'Month', 'Day', 'Date', 'Day of Week', 'Direction']
    data = data.groupby(col)['Volume'].sum().reset_index()
    # On va normaliser (méthode min-max) les valeurs
    volume_max, volume_min = data['Volume'].max(), data['Volume'].min()
    data = data.pivot_table(index=col_no_hour, columns='Hour', values='Volume').reset_index()
    data_pred = data_pred.groupby(col)['Volume'].sum().reset_index()
    data_pred = data_pred.pivot_table(index=col_no_hour, columns='Hour', values='Volume').reset_index()
    # Suppression des jours contenant des données manquantes
    data = data.dropna()
    data_pred = data_pred.dropna()
    # On garde les valeurs de mois entre 0 et 11 (plutôt que 1 et 12), ce qui sera plus pratique pour créer des one-hot vectors
    data['Month'] = data['Month'] - 1
    data_pred['Month'] = data_pred['Month'] - 1

    data_for = []
    data_for_date = []
    data_for_hour = []
    forecast = []
    forecast_date = []
    # Pour chaque jour
    for _, row in data_pred.iterrows():
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
            data_for_hour += [(i + longueur_serie - 1)%24 for i in range(len(target))]
            for k in range(len(target)):
                t, s1, s2, s3 = target[k], serie_J[k], serie_J_moins_1[k], serie_J_moins_7[k]
                # On normalise les valeurs
                s1_norm = list((s1 - volume_min)/(volume_max - volume_min))
                s2_norm = list((s2 - volume_min)/(volume_max - volume_min))
                s3_norm = list((s3 - volume_min)/(volume_max - volume_min))
                t_norm = (t - volume_min)/(volume_max - volume_min)
                
                # On récupère les date et les vraies données pour plot
                data_for_date.append(date)
                data_for.append([latitude, longitude, month, day_week, direction] + s1_norm + s2_norm + s3_norm + [t_norm])
                
                # Si on a déjà prédit la journée, on l'utilise
                if date - pd.to_timedelta(1, unit='d') in forecast_date:
                    i = forecast_date.index(date - pd.to_timedelta(1, unit='d'))
                    l = forecast_date.count(forecast_date[i])
                    s1_norm = forecast[i:i + l - 1]
                if date - pd.to_timedelta(2, unit='d') in forecast_date:
                    i = forecast_date.index(date - pd.to_timedelta(2, unit='d'))
                    l = forecast_date.count(forecast_date[i])
                    s2_norm = forecast[i:i + l]
                if date - pd.to_timedelta(8, unit='d') in forecast_date:
                    i = forecast_date.index(date - pd.to_timedelta(8, unit='d'))
                    l = forecast_date.count(forecast_date[i])
                    s3_norm = forecast[i:i + l]
                # On prédit
                output = model.forward(torch.tensor(latitude).unsqueeze(0), torch.tensor(longitude).unsqueeze(0), torch.eye(12)[int(round(month))].unsqueeze(0), torch.eye(7)[int(round(day_week))].unsqueeze(0), torch.eye(5)[int(round(direction))].unsqueeze(0), torch.tensor(s1_norm).unsqueeze(0), torch.tensor(s2_norm).unsqueeze(0), torch.tensor(s3_norm).unsqueeze(0))
                forecast.append(output.detach())
                forecast_date.append(date)

    return (np.array(data_for), np.array(data_for_date), np.array(data_for_hour), volume_max, volume_min, np.array(forecast))

def plot(data_for, output, data_for_date, data_for_hour):
    '''
    Plot et renvoie du DataFrame avec les données prédites et vraies

    Parameters
    ----------
    data_for : TYPE
        DESCRIPTION.
    output : TYPE
        DESCRIPTION.
    data_for_date : TYPE
        DESCRIPTION.
    data_for_hour : TYPE
        DESCRIPTION.

    '''
    data_for = data_for[:,[0,1,2,3,4,-1]]
    data_for_pd = pd.DataFrame(data_for)
    data_for_pd.columns = ['latitude', 'longitude', 'month', 'day_week', 'direction', 'to_pred']
    data_for_datetime = [data_for_date[i] + pd.to_timedelta(data_for_hour[i], unit='h') for i in range(len(data_for_date))]
    data_for_pd['Datetime'] = data_for_datetime
    data_for_pd.index = data_for_pd['Datetime']
    data_for_pd['pred'] = output
    plt.figure(0)
    data_for_pd['to_pred'].plot(label='Data')
    data_for_pd['pred'].plot(label='Pred')
    plt.ylabel("Volume")
    plt.title("Data vs Pred")
    plt.legend()
    plt.show()
    return data_for_pd