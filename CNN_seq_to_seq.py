# -*- coding: utf-8 -*-
"""
Created on Sat Nov 28 23:02:46 2020

@author: Patrice CHANOL & Corentin MORVAN--CHAUMEIL
"""

import torch
import torch.nn as nn

# %% Model


class CNN(nn.Module):
    """
    Implémentation d'un CNN seq-to-seq.

    Classe du modèle CNN final utilisé
    """

    def __init__(self, input_window, output_window):
        """
        Init.

        Parameters
        ----------
        input_window: int
            Représente le nombre de jour de la séquence d'entrée
            Longueur de la séquence d'entrée: 24 * input_window
        output_window: int
            Représente le nombre d'heure de la séquence de sortie
            Longueur de la séquence de sortie: output_window
        """
        super(CNN, self).__init__()

        self.input_window = input_window
        self.output_window = output_window
        self.name_model = "CNN"

        self.layer1 = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=24, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2)
        )
        self.layer2 = nn.Sequential(
            nn.Conv1d(in_channels=24, out_channels=48, kernel_size=3, padding=1),
            nn.BatchNorm1d(48),
            nn.ReLU(),
            nn.AdaptiveMaxPool1d(self.input_window * 24)
        )
        self.fc1 = nn.Linear(in_features=48 * (self.input_window * 24) + 7, out_features=128)
        self.fc2 = nn.Linear(in_features=128, out_features=32)
        self.fc3 = nn.Linear(in_features=32, out_features=self.output_window)

    def forward(self, day_of_week, serie_input):
        """Forward Pass."""
        serie_input = serie_input.float().unsqueeze(1)
        day_of_week = day_of_week.float()

        input_1 = serie_input

        out = self.layer1(input_1)
        out = self.layer2(out)
        out = out.view(out.size(0), -1)

        input_2 = torch.cat((out, day_of_week), dim=1)

        out = self.fc1(input_2)
        out = out.relu()
        out = nn.Dropout(0.25)(out)
        out = self.fc2(out)
        out = out.relu()
        out = self.fc3(out)
        return out

    def save(self):
        """Enregistre le modèle pour inférence dans le futur."""
        torch.save(self.state_dict(), './models/model_' + self.name_model + '_' + str(self.input_window) + '_days_to_' + str(self.output_window) + '_hours.pt')

    def load(self):
        """Récupère un modèle déjà entrainé pour inférer."""
        self.load_state_dict(torch.load('./models/model_' + self.name_model + '_' + str(self.input_window) + '_days_to_' + str(self.output_window) + '_hours.pt'))

# Our input timeserie is changing in a following way:

#     1st Convolution layer : (input_window*24) * 1, output: (input_window*24-2) * 24
#     1st Max Pooling layer : (input_window*24-2) * 24, output: (input_window*24-2)//2 * 24
#     2nd Convolution layer : (input_window*24-2)//2 * 24, output: (input_window*24-2)//2 - 2 * 48
#     2nd Adaptive Max Pooling layer : (input_window*24-2)//2 - 2 * 48, output: (input_window*24) * 48

#     First Linear layer : input : 48*(input_window*24) + 7, output: 128
#     Second Linear layer : input :128, output: 32
#     Third Linear layer : input : 32, output: output_window

# %% Previous Model

class CNN_classical(nn.Module):
    """Implémentation d'un CNN classique seq-to-seq."""

    def __init__(self, input_window, output_window):
        """
        Init.

        Parameters
        ----------
        input_window: int
            Représente le nombre de jour de la séquence d'entrée
            Longueur de la séquence d'entrée: 24 * input_window
        output_window: int
            Représente le nombre d'heure de la séquence de sortie
            Longueur de la séquence de sortie: output_window
        """
        super(CNN_classical, self).__init__()

        self.input_window = input_window
        self.output_window = output_window
        self.name_model = "CNN_classical"

        self.layer1 = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=24, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2)
        )
        self.layer2 = nn.Sequential(
            nn.Conv1d(in_channels=24, out_channels=48, kernel_size=3, padding=1),
            nn.BatchNorm1d(48),
            nn.ReLU(),
            nn.AdaptiveMaxPool1d(self.input_window * 24)
        )
        self.fc1 = nn.Linear(in_features=48*(self.input_window * 24), out_features=128)
        self.fc2 = nn.Linear(in_features=128, out_features=32)
        self.fc3 = nn.Linear(in_features=32, out_features=self.output_window)

    def forward(self, day_of_week, serie_input):
        """Forward Pass."""
        serie_input = serie_input.float().unsqueeze(1)
        day_of_week = day_of_week.float()

        input_1 = serie_input

        out = self.layer1(input_1)
        out = self.layer2(out)
        out = out.view(out.size(0), -1)

        out = self.fc1(out)
        out = out.relu()
        out = nn.Dropout(0.25)(out)
        out = self.fc2(out)
        out = out.relu()
        out = self.fc3(out)
        return out

    def save(self):
        """Enregistre le modèle pour inférence dans le futur."""
        torch.save(self.state_dict(), './models/model_' + self.name_model + '_' + str(self.input_window) + '_days_to_' + str(self.output_window) + '_hours.pt')

    def load(self):
        """Récupère un modèle déjà entrainé pour inférer."""
        self.load_state_dict(torch.load('./models/model_' + self.name_model + '_' + str(self.input_window) + '_days_to_' + str(self.output_window) + '_hours.pt'))

class CNN_3conv(nn.Module):
    """Implémentation d'un CNN complexe seq-to-seq."""

    def __init__(self, input_window, output_window):
        """
        Init.

        Parameters
        ----------
        input_window: int
            Représente le nombre de jour de la séquence d'entrée
            Longueur de la séquence d'entrée: 24 * input_window
        output_window: int
            Représente le nombre d'heure de la séquence de sortie
            Longueur de la séquence de sortie: output_window
        """
        super(CNN_3conv, self).__init__()

        self.input_window = input_window
        self.output_window = output_window
        self.name_model = "CNN_3conv"

        self.layer_tot_1 = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=24, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2)
        )
        self.layer_tot_2 = nn.Sequential(
            nn.Conv1d(in_channels=24, out_channels=48, kernel_size=3),
            nn.ReLU(),
            nn.AdaptiveMaxPool1d(self.input_window * 24)
        )

        self.layer_half_1 = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=24, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2)
        )
        self.layer_half_2 = nn.Sequential(
            nn.Conv1d(in_channels=24, out_channels=48, kernel_size=3),
            nn.ReLU(),
            nn.AdaptiveMaxPool1d(self.input_window * 24)
        )

        self.layer_quart_1 = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=24, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2)
        )
        self.layer_quart_2 = nn.Sequential(
            nn.Conv1d(in_channels=24, out_channels=48, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveMaxPool1d(self.input_window * 24)
        )

        self.layer2 = nn.Sequential(
            nn.Conv1d(in_channels=48, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.AdaptiveMaxPool1d(self.input_window * 24)
        )
        self.fc1 = nn.Linear(in_features=64*(self.input_window * 24) + 7, out_features=128)
        self.fc2 = nn.Linear(in_features=128, out_features=32)
        self.fc3 = nn.Linear(in_features=32, out_features=self.output_window)

    def forward(self, day_of_week, serie_input):
        """Forward Pass."""
        serie_input = serie_input.float().unsqueeze(1)
        day_of_week = day_of_week.float()

        input_1_tot = serie_input
        input_1_half = nn.functional.interpolate(input_1_tot, input_1_tot.shape[-1]//2)
        input_1_quart = nn.functional.interpolate(input_1_tot, input_1_tot.shape[-1]//4)

        out_tot = self.layer_tot_1(input_1_tot)
        out_tot = self.layer_tot_2(out_tot)
        out_half = self.layer_half_1(input_1_half)
        out_half = self.layer_half_2(out_half)
        out_quart = self.layer_quart_1(input_1_quart)
        out_quart = self.layer_quart_2(out_quart)

        out = torch.cat((out_tot, out_half, out_quart), dim=2)
        out = self.layer2(out)
        out = out.view(out.size(0), -1)

        input_2 = torch.cat((out, day_of_week), dim=1)

        out = self.fc1(input_2)
        out = out.relu()
        out = nn.Dropout(0.25)(out)
        out = self.fc2(out)
        out = out.relu()
        out = self.fc3(out)

        return out.view(-1)
