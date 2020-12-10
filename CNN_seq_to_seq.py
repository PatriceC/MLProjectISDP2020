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
        Implémentation d'un CNN seq-to-seq
        Classe du modèle CNN final utilisé
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
            nn.AdaptiveMaxPool1d(input_window * 24)
        )
        self.fc1 = nn.Linear(in_features=48 * (input_window * 24) + 7, out_features=128)
        self.fc2 = nn.Linear(in_features=128, out_features=32)
        self.fc3 = nn.Linear(in_features=32, out_features=output_window)

    def forward(self, day_of_week, serie_input):

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
        """
            Enregistre le modèle pour inférence dans le futur
        """
        torch.save(self.state_dict(), './models/model_' + self.name_model + '_' + str(self.input_window) + '_days_to_' + str(self.output_window) + '_hours.pt')
    
    def load(self):
        """
            Récupère un modèle déjà entrainé pour inférer
        """
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

    def __init__(self, S):
        super(CNN_classical, self).__init__()

        self.layer1 = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=24, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2)
        )
        self.layer2 = nn.Sequential(
            nn.Conv1d(in_channels=24, out_channels=48, kernel_size=3, padding=1),
            nn.BatchNorm1d(48),
            nn.ReLU(),
            nn.AdaptiveMaxPool1d(S-1)
        )
        self.fc1 = nn.Linear(in_features = 48*(S-1), out_features=128)
        self.fc2 = nn.Linear(in_features=128, out_features=32)
        self.fc3 = nn.Linear(in_features=32, out_features=1)

    def forward(self, x_latitude, x_longitude, x_month, x_day_week, x_direction, x_1, x_2, x_3):
        out = self.layer1(x_1.double().unsqueeze(1))
        out = self.layer2(out)
        out = out.view(out.size(0), -1)
        out = self.fc1(out)
        out = out.relu()
        out = nn.Dropout(0.25)(out)
        out = self.fc2(out)
        out = out.relu()
        out = self.fc3(out)

        return out.view(-1)
  
# Our input timeserie is changing in a following way:

#     1st Convolution layer : (S-1) * 1, output: (S-3) * 24
#     1st Max Pooling layer : (S-3) * 24, output: (S-3)//2 * 24
#     2nd Convolution layer : (S-1)//2 * 24, output: (S-3) * 48
#     2nd Adaptive Max Pooling layer : (S-3)//2 * 48, output: (S-1) * 48

#     First Linear layer : input : 48*(S-1), output: 128
#     Second Linear layer : input :128, output: 32
#     Third Linear layer : input : 32, output: 1

class CNN_3conv(nn.Module):

    def __init__(self, S):
        super(CNN_3conv, self).__init__()

        self.layer_tot_1 = nn.Sequential(
            nn.Conv1d(in_channels=3, out_channels=24, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2)
        )
        self.layer_tot_2 = nn.Sequential(
            nn.Conv1d(in_channels=24, out_channels=48, kernel_size=3),
            nn.ReLU(),
            nn.AdaptiveMaxPool1d(S-1)
        )

        self.layer_half_1 = nn.Sequential(
            nn.Conv1d(in_channels=3, out_channels=24, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2)
        )
        self.layer_half_2 = nn.Sequential(
            nn.Conv1d(in_channels=24, out_channels=48, kernel_size=3),
            nn.ReLU(),
            nn.AdaptiveMaxPool1d(S-1)
        )

        self.layer_quart_1 = nn.Sequential(
            nn.Conv1d(in_channels=3, out_channels=24, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2)
        )
        self.layer_quart_2 = nn.Sequential(
            nn.Conv1d(in_channels=24, out_channels=48, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveMaxPool1d(S-1)
        )

        self.layer2 = nn.Sequential(
            nn.Conv1d(in_channels=48, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.AdaptiveMaxPool1d(S-1)
        )
        self.fc1 = nn.Linear(in_features = 64*(S-1) + 2 + 1 +1 + 12 + 7 + 5, out_features=128)
        self.fc2 = nn.Linear(in_features=128, out_features=32)
        self.fc3 = nn.Linear(in_features=32, out_features=1)

    def forward(self, x_latitude, x_longitude, x_month, x_day_week, x_direction, x_1, x_2, x_3):
        x_tot = torch.cat((x_1.double().unsqueeze(1), x_2[:,:-1].double().unsqueeze(1), x_3[:,:-1].double().unsqueeze(1)), dim=1)
        x_half = nn.functional.interpolate(x_tot, x_tot.shape[-1]//2)
        x_quart = nn.functional.interpolate(x_tot, x_tot.shape[-1]//4)
        out_tot = self.layer_tot_1(x_tot)
        out_tot = self.layer_tot_2(out_tot)
        out_half = self.layer_half_1(x_half)
        out_half = self.layer_half_2(out_half)
        out_quart = self.layer_quart_1(x_quart)
        out_quart = self.layer_quart_2(out_quart)
        out = torch.cat((out_tot, out_half, out_quart), dim=2)
        out = self.layer2(out)
        out = out.view(out.size(0), -1)
        out = torch.cat((out, x_2[:,-1].double().unsqueeze(1), x_3[:,-1].double().unsqueeze(1), x_latitude.unsqueeze(1), x_longitude.unsqueeze(1), x_month, x_day_week, x_direction), dim=1)
        out = self.fc1(out)
        out = out.relu()
        out = nn.Dropout(0.25)(out)
        out = self.fc2(out)
        out = out.relu()
        out = self.fc3(out)

        return out.view(-1)

# Our input timeserie is changing in a following way:

#     Local Convulation 1st layer Total : input : (S-1) * 3, output: (S-1) * 24
#       Local Max Pooling 1st layer Total : input : (S-1) * 24, output: (S-1)//2 * 24
#       Local Convulation 2nd layer Total : input : (S-1)//2 * 24, output: (S-1)//2-2 * 48
#           Local Adaptative Max Pooling 2nd layer Total : input : (S-1)//2-2 * 48, output: (S-1) * 48
#     Local Convulation 1st layer Half : input : (S-1)//2 * 3, output: (S-1)//2 * 24
#       Local Max Pooling 1st layer Half : input : (S-1)//2 * 24, output: (S-1)//4 * 24
#       Local Convulation 2nd layer Half : input : (S-1)//4 * 24, output: (S-1)//4-2 * 48
#           Local Adaptative Max Pooling 2nd layer Half : input : (S-1)//4-2 * 48, output: (S-1) * 48
#     Local Convulation 1st layer Quart : input : (S-1)//4 * 3, output: (S-1)//4 * 24
#       Local Max Pooling 1st layer Quart : input : (S-1)//4 * 24, output: (S-1)//8 * 24
#       Local Convulation 2nd layer Quart : input : (S-1)//8 * 24, output: (S-1)//8 * 48
#           Local Adaptative Max Pooling 2nd layer Quart : input : (S-1)//8 * 24, output: (S-1) * 48

#     Concat of all the Local Conv layers
#     Convolution layer : input : (S-1) * 48, output: (S-3) * 64
#     Adaptive Max Pooling layer : input : (S-3) * 64, output: (S-1) * 64

#     First Linear layer : input : 64*(S-1) + 2 + 1 +1 + 12 + 7 + 5, output: 128
#     Second Linear layer : input : 128, output: 32
#     Third Linear layer : input : 32, output: 1