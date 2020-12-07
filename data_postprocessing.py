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

def process_data(date_range=['2017','2020'], direction=None, latitude=[-100,100], longitude=[-100,100], longueur_serie=6, file='./Radar_Traffic_Counts.csv'):
    '''
    Génération du Dataset désiré de post

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

    '''

    data = pd.read_csv(file)

    # On va normaliser (méthode min-max) les valeurs de latitude et longitude
    latitude_max, latitude_min = data['location_latitude'].max(), data['location_latitude'].min()
    longitude_max, longitude_min = data['location_longitude'].max(), data['location_longitude'].min()

    # Préparation du Dataset
    data = data.drop(columns=['location_name', 'Time Bin'])
    data['Direction'] = data['Direction'].astype('category').cat.codes
    data['Date'] = pd.to_datetime(data[['Year', 'Month', 'Day']], errors = 'coerce')
    data = data.sort_values(['Year', 'Month', 'Day', 'Hour'])
    data_pred = data[(data['Date'] >= date_range[0]) & (data['Date'] <= date_range[1])]
    data_pred = data_pred[(data_pred['location_latitude'] >= latitude[0]) & (data_pred['location_latitude'] <= latitude[1])]
    data_pred = data_pred[(data_pred['location_longitude'] >= longitude[0]) & (data_pred['location_longitude'] <= longitude[1])]
    if direction != None:
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

    data_post = []
    data_post_date = []
    data_post_hour = []
    # Pour chaque jour
    for _, row in data_pred.iterrows():
        # On récuprère les informations de cette donnée
        latitude, longitude = row['location_latitude'], row['location_longitude']
        month, day_week, date = row['Month'], row['Day of Week'], row['Date']
        direction = row['Direction']
        # On génère les séries
        result = series(Date_J=date, latitude=latitude, longitude=longitude, direction=direction, longueur_serie=longueur_serie, data=data)

        if result is not None:
            # On normalise (méthode min-max) les valeurs de latitude et longitude
            latitude = (latitude - latitude_min)/(latitude_max - latitude_min)
            longitude = (longitude - longitude_min)/(longitude_max - longitude_min)
            target, serie_J, serie_J_moins_1, serie_J_moins_7 = result
            # On récupère les heures pour plot
            data_post_hour += [i for i in range(len(target))]
            for t, s1, s2, s3 in zip(target, serie_J, serie_J_moins_1, serie_J_moins_7):
                # On normalise les valeurs
                s1_norm = list((s1 - volume_min)/(volume_max - volume_min))
                s2_norm = list((s2 - volume_min)/(volume_max - volume_min))
                s3_norm = list((s3 - volume_min)/(volume_max - volume_min))
                t_norm = (t - volume_min)/(volume_max - volume_min)
                # On récupère les date et les vraies données
                data_post_date.append(date)
                data_post.append([latitude, longitude, month, day_week, direction] + s1_norm + s2_norm + s3_norm + [t_norm])

    return (np.array(data_post), np.array(data_post_date), np.array(data_post_hour), volume_max, volume_min)

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
    
    month_post = np.zeros((n_post, 12))
    day_week_post = np.zeros((n_post, 7))
    direction_post = np.zeros((n_post, 5))
    
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

def data_pred(data_loader_post, model):
    # Prediction
    (latitude, longitude, month, day_week, direction, serie_J, serie_J_moins_1, serie_J_moins_7), _ = next(iter(data_loader_post))
    return model.forward(latitude, longitude, month, day_week, direction, serie_J, serie_J_moins_1, serie_J_moins_7).view(-1)

def plot(data_post, output, data_post_date, data_post_hour):
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
    data_post = data_post[:,[0,1,2,3,4,-1]]
    data_post_pd = pd.DataFrame(data_post)
    data_post_pd.columns = ['latitude', 'longitude', 'month', 'day_week', 'direction', 'to_pred']
    data_post_datetime = [data_post_date[i] + pd.to_timedelta(data_post_hour[i], unit='h') for i in range(len(data_post_date))]
    data_post_pd['Datetime'] = data_post_datetime
    data_post_pd.index = data_post_pd['Datetime']
    data_post_pd['pred'] = output
    localisation = data_post_pd[['latitude','longitude']].drop_duplicates().to_numpy()
    for loc in range(len(localisation)):
        plt.figure(loc+1)
        data_post_pd[(data_post_pd['latitude'] == localisation[loc, 0]) & (data_post_pd['longitude'] == localisation[loc, 1])]['to_pred'].plot(label='Data')
        data_post_pd[(data_post_pd['latitude'] == localisation[loc, 0]) & (data_post_pd['longitude'] == localisation[loc, 1])]['pred'].plot(label='Pred')
        plt.ylabel("Volume")
        plt.title("Data vs Pred")
        plt.legend()
        plt.show()
    return data_post_pd