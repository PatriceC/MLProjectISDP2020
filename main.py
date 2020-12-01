# -*- coding: utf-8 -*-
"""
Created on Mon Nov  9 12:49:11 2020

@author: Patrice CHANOL & Corentin MORVAN--CHAUMEIL
"""

import numpy as np
import torch as torch
import torch.nn as nn 

import preprocessing_data
import model_training
import data_postprocessing

import LSTM
import CNN

# %% Data Preprocessing

longueur_serie = 12

data_input = input("Avez-vous déjà les fichiers {} et {} ? [O/N]\n".format('data_train_' + str(longueur_serie) + '.txt', 'data_test_' + str(longueur_serie) + '.txt'))

if data_input != 'O':
    ## On crée un dataset train/test pour l'apprentissage de notre modèle
    data_train, data_test = preprocessing_data.process_data(longueur_serie=longueur_serie)
else:
    # Ou alors on récupère un dataset déjà créé
    data_train = np.genfromtxt('./data_train_' + str(longueur_serie) + '.txt')
    data_test = np.genfromtxt('./data_test_' + str(longueur_serie) + '.txt')

n_train, n_test = data_train.shape[0], data_test.shape[0]
#data_train, data_test = data_train[:int(0.3*n_train)], data_test[:int(0.3*n_test)]

batch_size = 128

data_loader_train, data_loader_test = preprocessing_data.data_loader(data_train, data_test, longueur_serie, batch_size = 128)

# %% Définition du model utilisé

model_dispo = ['LSTM', 'CNN']
nom_model = input("Choisir le modèle à traiter parmis : {}\n".format(model_dispo))

if nom_model == 'LSTM':
    model = LSTM.LSTM_NN(longueur_serie=longueur_serie)
    print(model)
    error = nn.L1Loss()
    learning_rate = 0.001
    weight_decay = 0.0001
    lr_dim = 2
    num_epoch = 1
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
elif nom_model == 'CNN':
    model = CNN.CNN(S=longueur_serie).double()
    print(model)
    error = nn.L1Loss()
    learning_rate = 0.01
    weight_decay = 0.0001
    lr_dim = 10
    num_epoch = 2
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

else:
    print("Erreur dans le choix du modèle")

# %% Training and Testing

model, pourcentage_loss_list, test_loss_list = model_training.main(nom_model, model, error, data_loader_train, data_loader_test, n_train, learning_rate, lr_dim, weight_decay, num_epoch, batch_size)

# %% Validation

data_post, data_post_date, volume_max, volume_min = data_postprocessing.process_data(date_range=['2018-07-01','2018-07-30'], direction=0, longueur_serie=longueur_serie)
data_loader_post = data_postprocessing.data_loader(data_post, longueur_serie)
output = data_postprocessing.data_pred(data_loader_post, model)

data_post[:,-1] = data_post[:,-1]*(volume_max - volume_min) + volume_min
output = output*(volume_max - volume_min) + volume_min

data_post_pd = data_postprocessing.plot(data_post, output.detach(), data_post_date)