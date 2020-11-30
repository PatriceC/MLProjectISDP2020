# -*- coding: utf-8 -*-
"""
Created on Sat Nov 28 23:02:46 2020

@author: Patrice CHANOL
"""

import torch
import torch.nn as nn


# %% Model

class CNN(nn.Module):

    def __init__(self, S):
        super(CNN, self).__init__()

        self.layer_1 = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=6, kernel_size=3),
            nn.Tanh(),
            nn.MaxPool1d(kernel_size=2)
        )

        self.layer_2 = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=6, kernel_size=3),
            nn.Tanh(),
            nn.MaxPool1d(kernel_size=2)
        )

        self.layer_3 = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=6, kernel_size=3),
            nn.Tanh(),
            nn.MaxPool1d(kernel_size=2)
        )

        if S%2:
            fc1_features = int(((3*S-9)/2 * 6)) + 2 + 12 + 7 + 5
        else:
            fc1_features = int(((3*S-8)/2 * 6)) + 2 + 12 + 7 + 5
        self.fc1 = nn.Linear(in_features = fc1_features, out_features=64)
        self.fc2 = nn.Linear(in_features=64, out_features=32)
        self.fc3 = nn.Linear(in_features=32, out_features=16)
        self.fc4 = nn.Linear(in_features=16, out_features=1)

    def forward(self, x_latitude, x_longitude, x_month, x_day_week, x_direction, x_1, x_2, x_3):
        out_1 = self.layer_1(x_1.double())
        out_1 = out_1.view(out_1.size(0), -1)
        out_2 = self.layer_2(x_2.double())
        out_2 = out_2.view(out_2.size(0), -1)
        out_3 = self.layer_3(x_3.double())
        out_3 = out_3.view(out_3.size(0), -1)
        out = torch.cat((out_1, out_2, out_3), dim=1)
        out = torch.cat((out, x_latitude.double(), x_longitude.double(), x_month.double(), x_day_week.double(), x_direction.double()), dim=1)
        out = out.view(out.size(0), -1)
        out = self.fc1(out)
        out = out.relu()
        out = nn.Dropout(0.25)(out)
        out = self.fc2(out)
        out = out.relu()
        out = self.fc3(out)
        out = out.relu()
        out = self.fc4(out)

        return out.view(-1)

# Our input timeserie is changing in a following way:
# Note that S is even or odd : Se or So
#     Local Convulation layer 1st Serie input : input: (S-1) * 1, output: (S-3) * 6
#     Local Max Pooling layer 1st Serie input: input: (S-3) * 6, output: (So-3)/2 * 6 or (Se-4)/2 * 6
#     Local Convulation layer 2nd Serie input: input: S * 1, output: (S-2) * 6
#     Local Max Pooling layer 2nd Serie input: input: (S-2) * 6, output: (Se-2)/2 * 6 or (So-3)/2 * 6
#     Local Convulation layer 3rd Serie input: input: S * 1, output: (S-2) * 6
#     Local Max Pooling layer 3rd Serie input: input: (S-2) * 6, output: (Se-2)/2 * 6 or (So-3)/2 * 6
#     Concat of all the Local Conv layers
#     First Linear layer : input: ((3*Se-8)/2 * 6) + 2 + 12 + 7 + 5 or ((3*So-9)/2 * 6) + 2 + 12 + 7 + 5, output: 64
#     Second Linear layer : input: 64, output: 32
#     Third Linear layer : input: 32, output: 16
#     Fourth Linear layer : input: 16, output: 1
