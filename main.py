# -*- coding: utf-8 -*-
"""
Created on Mon Nov  9 12:49:11 2020

@author: Patrice CHANOL & Corentin MORVAN--CHAUMEIL
"""

import torch as torch
import torch.nn as nn
from sys import exit

import data_preprocessing
import model_training
import visualisation

import LSTM_seq_to_seq
import CNN_seq_to_seq
import TRANSFORMER_seq_to_seq

# %% Data Preprocessing

# Nombre de jours en input_window, Nombre d'heures en output_window,
input_window, output_window = 7, 24

data_input = input("Avez-vous déjà les fichiers {} et {} ? [O/N]\n".format('data/data_train_{}_days_to_{}_hours.txt'.format(input_window, output_window), 'data/data_test_{}_days_to_{}_hours.txt'.format(input_window, output_window)))

if data_input == 'O':
    try:
        # Ou alors on récupère un dataset déjà créé
        data_train = torch.load('./data/data_train_{}_days_to_{}_hours.txt'.format(input_window, output_window))
        data_test = torch.load('./data/data_test_{}_days_to_{}_hours.txt'.format(input_window, output_window))
    except FileNotFoundError:
        print("Pas de fichiers trouvés")
        continuer = input('Souhaitez-vous générer le dataset ? [O/N]\n')
elif data_input != 'O' or continuer == 'O':
    # On crée un dataset train/test pour l'apprentissage de notre modèle
    data_train, data_test = data_preprocessing.process_data(input_window, output_window)
else:
    exit()

batch_size = 256

data_train_loader, data_test_loader = data_preprocessing.data_loader(data_train, data_test, input_window, output_window, batch_size=batch_size)

# %% Définition du model utilisé

model_dispo = ['LSTM', 'CNN', 'Transformer']
nom_model = input("Choisir le modèle à traiter parmis : {}\n".format(model_dispo))

if nom_model == 'LSTM':
    model = LSTM_seq_to_seq.LSTM(input_window, output_window)
    print(model)
    criterion = nn.MSELoss()
    learning_rate = 0.002
    weight_decay = 0.0001
    num_epochs = 10
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1.0, gamma=0.50)
elif nom_model == 'CNN':
    model = CNN_seq_to_seq.CNN(input_window, output_window)
    print(model)
    criterion = nn.MSELoss()
    learning_rate = 0.002
    weight_decay = 0.0001
    num_epochs = 10
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1.0, gamma=0.50)
elif nom_model == 'Transformer':
    model = TRANSFORMER_seq_to_seq.Transformer(input_window, output_window)
    print(model)
    criterion = nn.MSELoss()
    learning_rate = 0.002
    weight_decay = 0.0001
    num_epochs = 10
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1.0, gamma=0.50)
else:
    exit("Erreur dans le choix du modèle")

# %% Training and Testing

trained = input('Souhaitez-vous charger un modèle pré-entrainé avec les mêmes paramètres ? [O/N]\n')
if trained == 'O':
    try:
        model.load()
    except FileNotFoundError:
        print("Pas de modèle pré-entrainé")
        continuer = input('Souhaitez-vous entrainer le modèle ? [O/N]\n')
elif trained != 'O' or continuer == 'O':
    model, test_loss_list = model_training.main(model, criterion, optimizer, scheduler, data_train_loader, data_test_loader, num_epochs, input_window, output_window, batch_size)
else:
    exit()

# %% Validation

# On va afficher les prédictions et les réalisations sur période fixée
# et une localisation et une direction fixée

visualisation.pred_vs_reality(model, input_window, output_window)

# %% Forecast

# On va afficher un forecast des données et les réalisations
# à partir d'une date fixée et une localisation et une direction fixée

visualisation.forecast(model, input_window, output_window)
