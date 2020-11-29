# -*- coding: utf-8 -*-
"""
Created on Mon Nov 23 13:54:58 2020

@author: Patrice CHANOL
"""

import numpy as np
import pandas as pd
import torch

# %% Load Data

data_load = pd.read_csv('D:/Mines/3A/ML/Projet/archive/Radar_Traffic_Counts.csv')
print(data_load.head(5))
data_load = data_load.drop(columns=['Time Bin','location_name'])
data_load['Direction'] = data_load['Direction'].astype('category').cat.codes
print(data_load.head(5))

# %% Select set

def main(data=data_load, time_thin='Hour', series=6, pred=1, batch_size=128, train_per_length=0.8):
    """
    Fonction main : renvoie un objet dataloader avec un train_set et un test_set
    pour une time serie donnée

    Parameters
    ----------
    data : TYPE, optional
        DESCRIPTION. The default is data_load.
    time_thin : TYPE, optional
        DESCRIPTION. The default is 'Hour'.
    series : TYPE, optional
        DESCRIPTION. The default is 6.
    pred : TYPE, optional
        DESCRIPTION. The default is 1.
    batch_size : TYPE, optional
        DESCRIPTION. The default is 128.
    train_per_length : TYPE, optional
        DESCRIPTION. The default is 0.8.

    Raises
    ------
    ValueError
        DESCRIPTION.

    Returns
    -------
    train_set : TYPE
        DESCRIPTION.
    test_set : TYPE
        DESCRIPTION.

    """
    if time_thin == 'Hour':
        col = ['Direction', 'location_latitude', 'location_longitude',
               'Year', 'Month', 'Day of Week', 'Day']
        data_pd = data.groupby(col + [time_thin])['Volume'].sum().reset_index()
        data_pd = data.pivot_table(index=col, columns=time_thin,
                                    values='Volume', fill_value=0).reset_index()
    elif time_thin == 'Day':
        col = ['Direction', 'location_latitude', 'location_longitude',
               'Year', 'Month', 'Day of Week']
        data_pd = data.groupby(col + [time_thin])['Volume'].sum().reset_index()
        data_pd = data.pivot_table(index=col, columns=time_thin,
                                    values='Volume', fill_value=0).reset_index()
    elif time_thin == 'Month':
        col = ['Direction', 'location_latitude', 'location_longitude',
               'Year']
        data_pd = data.groupby(col + [time_thin])['Volume'].sum().reset_index()
        data_pd = data.pivot_table(index=col, columns=time_thin,
                                    values='Volume', fill_value=0).reset_index()
    elif time_thin == 'Year':
        col = ['Direction', 'location_latitude', 'location_longitude']
        data_pd = data.groupby(col + [time_thin])['Volume'].sum().reset_index()
        data_pd = data.pivot_table(index=col, columns=time_thin,
                                    values='Volume', fill_value=0).reset_index()
    
    shape = data_pd.shape
    c = len(col)
    if c + series + pred > shape[1]:
        raise ValueError('Taille de la série et du moment à prédire trop grand')
    nb_series = shape[1] - pred - series - c
    data_set = torch.empty(shape[0]*nb_series, c - 1 + series + pred)
    direction_set = torch.empty(shape[0]*nb_series, 5)
    D = list(data_pd.iterrows())
    for k in range(shape[0]):
        temp = D[k][1].to_list()
        for i in range(nb_series):
            direction_set[k*nb_series + i] = torch.tensor([int(temp[0] == j) for j in range(5)])
            data_set[k*nb_series + i] = torch.tensor(temp[1:c] + temp[c + i : c + i + series + pred])
    n_train = int(shape[0]*nb_series*train_per_length)
    perm = torch.randperm(shape[0]*nb_series)
    train_set = torch.utils.data.DataLoader(
        list(zip(zip(direction_set[perm[:n_train]], data_set[perm[:n_train],:-pred]),
                 data_set[perm[:n_train],-pred:])), batch_size=batch_size)
    test_set = torch.utils.data.DataLoader(
        list(zip(zip(direction_set[perm[n_train:]], data_set[perm[n_train:],:-pred]),
                 data_set[perm[n_train:],-pred:])), batch_size=batch_size)
    return train_set, test_set

a,b = main(data=data_load, time_thin='Hour', series=6, pred=1, batch_size=128, train_per_length=0.8)

compt = 0
for i in a:
    print(i)
    compt += 1
    if compt == 10:
        break