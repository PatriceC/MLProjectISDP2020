# -*- coding: utf-8 -*-
"""
Created on Mon Nov  9 12:49:11 2020

@author: Patrice CHANOL & Corentin MORVAN--CHAUMEIL
"""
import numpy as np

import torch as torch
import torch.nn as nn 

import preprocessing_data

longueur_serie = 6

## On crée un dataset train/test pour l'apprentissage de notre modèle
#data_train, data_test = preprocessing_data.process_data(longueur_serie=longueur_serie)

# Ou alors on récupère un dataset déjà créé
data_train = np.genfromtxt('./data_train_' + str(longueur_serie) + '.txt')
data_test = np.genfromtxt('./data_test_' + str(longueur_serie) + '.txt')
n_train, n_test = data_train.shape[0], data_test.shape[0]

##### On dénit construit nos DataLoaders de train/test que nous utiliserons pour itérer sur les données pour l'apprentissage de modèles.
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

batch_size = 128

data_loader_train = torch.utils.data.DataLoader(list(zip(zip(latitude_train, longitude_train, month_train, day_week_train, direction_train, serie_J_train, serie_J_moins_1_train, serie_J_moins_7_train), target_train)), batch_size= batch_size, shuffle=True)
data_loader_test = torch.utils.data.DataLoader(list(zip(zip(latitude_test, longitude_test, month_test, day_week_test, direction_test, serie_J_test, serie_J_moins_1_test, serie_J_moins_7_test), target_test)), batch_size= batch_size, shuffle=True)
##### Fin de la création des DataLoaders de train/test