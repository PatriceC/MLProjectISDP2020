# -*- coding: utf-8 -*-
"""
Created on Mon Nov  9 12:49:11 2020

@author: Patrice CHANOL & Corentin MORVAN--CHAUMEIL
"""

import numpy as np
import torch as torch
import torch.nn as nn 

import preprocessing_data
import data_processing
import LSTM


# %% Data Preprocessing

longueur_serie = 9

data_input = input("Avez-vous déjà les fichiers {} et {} ? [O/N]\n".format('data_train_' + str(longueur_serie) + '.txt', 'data_test_' + str(longueur_serie) + '.txt'))

if data_input != 'O':
    ## On crée un dataset train/test pour l'apprentissage de notre modèle
    data_train, data_test = preprocessing_data.process_data(longueur_serie=longueur_serie)
else:
    # Ou alors on récupère un dataset déjà créé
    data_train = np.genfromtxt('./data_train_' + str(longueur_serie) + '.txt')
    data_test = np.genfromtxt('./data_test_' + str(longueur_serie) + '.txt')    

n_train, n_test = data_train.shape[0], data_test.shape[0]

batch_size = 128

data_loader_train, data_loader_test = preprocessing_data.data_loader(data_train, data_test, longueur_serie, batch_size = 128)

# %% Définition du model utilisé

model_dispo = ('LSTM')
nom_model = input("Choisir le modèle à traiter parmis : {}\n".format(model_dispo))

if nom_model == 'LSTM':
    model = LSTM.LSTM_NN(longueur_serie=longueur_serie)
    print(model)
    error = nn.L1Loss()
    learning_rate = 0.001
    weight_decay = 0.0001
    lr_dim = 2
    num_epoch = 2
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
else:
    print("Erreur dans le choix du modèle")

# %% Training and Testing

model, error, pourcentage_loss_list, test_loss_list = data_processing.main(model, error, data_loader_train, data_loader_test, n_train, learning_rate, lr_dim, weight_decay, num_epoch, batch_size)
