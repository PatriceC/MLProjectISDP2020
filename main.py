# -*- coding: utf-8 -*-
"""
Created on Mon Nov  9 12:49:11 2020

@author: Patrice CHANOL & Corentin MORVAN--CHAUMEIL
"""

import numpy as np
import torch as torch
import torch.nn as nn

import data_preprocessing
import model_training
import data_postprocessing
import data_forecast

import LSTM_seq_to_seq
import CNN_seq_to_seq

# %% Data Preprocessing

# Nombre de jours en input_window, Nombre d'heures en output_window, 
input_window, output_window = 7, 24

data_input = input("Avez-vous déjà les fichiers {} et {} ? [O/N]\n".format('data/data_train_{}_days_to_{}_hours.txt'.format(input_window, output_window), 'data/data_test_{}_days_to_{}_hours.txt'.format(input_window, output_window)))

if data_input != 'O':
    # On crée un dataset train/test pour l'apprentissage de notre modèle
    data_train, data_test = data_preprocessing.process_data(input_window, output_window)
else:
    # Ou alors on récupère un dataset déjà créé
    data_train = torch.load('./data/data_train_{}_days_to_{}_hours.txt'.format(input_window, output_window))
    data_test = torch.load('./data/data_test_{}_days_to_{}_hours.txt'.format(input_window, output_window))

n_train, n_test = data_train.shape[0], data_test.shape[0]
# data_train, data_test = data_train[:int(0.3*n_train)], data_test[:int(0.3*n_test)]

batch_size = 128

data_train_loader, data_test_loader = data_preprocessing.data_loader(data_train, data_test, input_window, output_window, batch_size=batch_size)

# %% Définition du model utilisé

model_dispo = ['LSTM', 'CNN']
nom_model = input("Choisir le modèle à traiter parmis : {}\n".format(model_dispo))

if nom_model == 'LSTM':
    model = LSTM_seq_to_seq.LSTM(input_window, output_window)
    print(model)
    criterion = nn.MSELoss()
    learning_rate = 0.001
    weight_decay = 0.0001
    num_epochs = 5
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1.0, gamma=0.50)
elif nom_model == 'CNN':
    model = CNN_seq_to_seq.CNN(input_window, output_window)
    print(model)
    criterion = nn.MSELoss()
    learning_rate = 0.001
    weight_decay = 0.0001
    num_epochs = 5
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1.0, gamma=0.50)
else:
    print("Erreur dans le choix du modèle")

# %% Training and Testing

model, test_loss_list = model_training.main(model, criterion, optimizer, scheduler, data_train_loader, data_test_loader, num_epochs, input_window, output_window, batch_size)

# %% Validation

# # On va afficher les prédictions et les réalisations sur une date fixée

# (data_post, data_post_date, data_post_hour, volume_max, volume_min) = data_postprocessing.process_data(date_range=['2018-07-09', '2018-08-10'], direction=0, longueur_serie=longueur_serie)
# data_loader_post = data_postprocessing.data_loader(data_post, longueur_serie)
# output = data_postprocessing.data_pred(data_loader_post, model)

# data_post[:, -1] = data_post[:, -1]*(volume_max - volume_min) + volume_min
# output = output*(volume_max - volume_min) + volume_min

# data_post_pd = data_postprocessing.plot(data_post, output.detach(), data_post_date, data_post_hour)

# %% Forecast

# On va afficher un forecast des données et les réalisations à partir d'une date fixée

# (data_for, data_for_date, data_for_hour, volume_max, volume_min, forecast) = data_forecast.forecast(model, date_range=['2018-07-09', '2018-08-10'], direction=0, latitude=30.268652000000003, longitude=-97.759929, longueur_serie=longueur_serie)

# data_for[:, -1] = data_for[:, -1]*(volume_max - volume_min) + volume_min
# forecast = forecast*(volume_max - volume_min) + volume_min

# data_post_pd = data_forecast.plot(data_for, forecast, data_for_date, data_for_hour)
