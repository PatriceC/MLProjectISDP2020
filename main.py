# -*- coding: utf-8 -*-
"""
Created on Mon Nov  9 12:49:11 2020

@author: Patrice CHANOL & Corentin MORVAN--CHAUMEIL
"""
import numpy as np
import matplotlib.pyplot as plt

import torch as torch
import torch.nn as nn 

import preprocessing_data
import LSTM

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

##### Dénition du model utilisé
model = LSTM.LSTM_NN()
error = nn.MSELoss()
learning_rate = 0.01
weight_decay = 0.0001
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
#####

test_loss_list, pourcentage_loss_list = [], []

count, pourcentage = 0, 0.
for (latitude, longitude, month, day_week, direction, serie_J, serie_J_moins_1, serie_J_moins_7), target in data_loader_train:

    latitude = latitude.float().unsqueeze(1)
    longitude = longitude.float().unsqueeze(1)
    month = month.float()
    day_week = day_week.float()
    direction = direction.float()
    serie_J = serie_J.float().unsqueeze(1)
    serie_J_moins_1 = serie_J_moins_1.float().unsqueeze(1)
    serie_J_moins_7 = serie_J_moins_7.float().unsqueeze(1)
    target = target.float().unsqueeze(1)

    output = model.forward(latitude, longitude, month, day_week, direction, serie_J, serie_J_moins_1, serie_J_moins_7)
    loss = error(output, target)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    count += 128

    if count >= pourcentage * n_train:
        test_loss_batch = []

        for (latitude_t, longitude_t, month_t, day_week_t, direction_t, serie_J_t, serie_J_moins_1_t, serie_J_moins_7_t), target_t in data_loader_test:

            latitude_t = latitude_t.float().unsqueeze(1)
            longitude_t = longitude_t.float().unsqueeze(1)
            month = month.float()
            day_week = day_week.float()
            direction = direction.float()
            serie_J_t = serie_J_t.float().unsqueeze(1)
            serie_J_moins_1_t = serie_J_moins_1_t.float().unsqueeze(1)
            serie_J_moins_7_t = serie_J_moins_7_t.float().unsqueeze(1)
            target_t = target_t.float().unsqueeze(1)

            output_t = model.forward(latitude_t, longitude_t, month_t, day_week_t, direction_t, serie_J_t, serie_J_moins_1_t, serie_J_moins_7_t)
            loss_test = error(output_t, target_t).data.item()
            test_loss_batch.append(loss_test)

        test_loss = np.mean(test_loss_batch)
        print("Pourcentage:", 100*pourcentage, "%")
        print(test_loss)
        print()
        pourcentage_loss_list.append(100*pourcentage)
        test_loss_list.append(test_loss)
        pourcentage += 0.1

print(pourcentage_loss_list, test_loss_list)
plt.plot(pourcentage_loss_list, test_loss_list)
plt.xlabel("Pourcentage")
plt.ylabel("MSE Loss")
plt.title("Test Loss vs Pourcentage")
plt.show()
