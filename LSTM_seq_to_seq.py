# -*- coding: utf-8 -*-
"""
Created on Sat Nov 28 23:02:46 2020

@author: Patrice CHANOL & Corentin MORVAN--CHAUMEIL
"""

import torch as torch
import torch.nn as nn

class LSTM(nn.Module):
    """
        Implémentation d'un LSTM seq-to-seq
    """
    def __init__(self, input_window, output_window):
        """
            input_window: int
                Représente le nombre de jour de la séquence d'entrée
                Longueur de la séquence d'entrée: 24 * input_window
            output_window: int
                Représente le nombre d'heure de la séquence de sortie
                Longueur de la séquence de sortie: output_window
        """
        super(LSTM, self).__init__()

        self.input_window = input_window
        self.output_window = output_window
        self.name_model = "LSTM"

        self.lstm = nn.LSTM(input_size=1, hidden_size=200, num_layers=1, batch_first=True)

        self.dropout = nn.Dropout(0.1)

        self.lin = nn.Linear(in_features=200 + 7, out_features=72) # Couche linéaire pour la sortie du LSTM + one_hot du jour de la semaine
        self.lin2 = nn.Linear(in_features=72, out_features=self.output_window)

        self.relu = nn.ReLU()

    def forward(self, day_of_week, serie_input):

        serie_input = serie_input.float().unsqueeze(2)
        day_of_week = day_of_week.float()
        
        input_1 = serie_input
        # On passe notre série d'entrée dans le LSTM
        out, _ = self.lstm(input_1)

        # On ne garde que les valeurs du dernier hidden state
        out = out[:, -1, :]

        # On applique le dropout pour limiter le sur-apprentissage
        out = self.dropout(out)

        # On concatène la sortie du LSTM avec le one-hot vector représentant le jour de la semaine avant de passer dans les couches linéaires
        input_2 = torch.cat((out, day_of_week), dim=1)
        out = self.lin(input_2)
        out = self.relu(out)
        out = self.lin2(out)
        return(out)
    
    def save(self):
        """
            Enregistre le modèle pour inférence dans le futur
        """
        torch.save(self.state_dict(), './models/model_' + self.name_model + '_' + str(self.input_window) + '_days_to_' + str(self.output_window) + '_hours.pt')
    
    def load(self):
        """
            Récupère un modèle déjà entrainé pour inférer
        """
        self.load_state_dict(torch.load('./models/model_' + self.name_model + '_' + str(self.input_window) + '_days_to_' + str(self.output_window) + '_hours.pt'))