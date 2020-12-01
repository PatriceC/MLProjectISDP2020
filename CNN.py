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
            nn.Conv1d(in_channels=1, out_channels=32, kernel_size=3),
            nn.Tanh(),
            nn.MaxPool1d(kernel_size=2)
        )
        self.layer_2 = nn.Sequential(
            nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3),
            nn.Tanh(),
            nn.AdaptiveAvgPool1d(S-1)
        )

        self.fc1 = nn.Linear(in_features = 64*(S-1), out_features=128)
        self.fc2 = nn.Linear(in_features=128, out_features=32)
        self.fc3 = nn.Linear(in_features=32, out_features=1)

    def forward(self, x_latitude, x_longitude, x_month, x_day_week, x_direction, x_1, x_2, x_3):
        out = x_1.double().unsqueeze(1)
        out = self.layer_1(out)
        out = self.layer_2(out)
        out = out.view(out.size(0), -1)
        out = self.fc1(out)
        out = out.relu()
        out = nn.Dropout(0.25)(out)
        out = self.fc2(out)
        out = out.relu()
        out = self.fc3(out)

        return out.view(-1)

# Our input timeserie is changing in a following way:
# Note that S is even or odd : Se or So
#     Local Convulation 1st layer 1st Serie input : (S-1) * 1, output: (S-3) * 32
#     Local Max Pooling 1st layer 1st Serie input: (S-3) * 32, output: (So-3)/2 * 32 or (Se-4)/2 * 32
#     Local Convulation 2nd layer 1st Serie input: (So-3)/2 * 32 or (Se-4)/2 * 32, output: (So-3)/2 * 64 or (Se-4)/2 * 64
#     Local Adaptative Max Pooling 2nd layer 1st Serie input: (So-3)/2 * 64 or (Se-4)/2 * 64, output: (S-1) * 64
#     First Linear layer : input: (S-1)*64, output: 128
#     Second Linear layer : input: 128, output: 32
#     Third Linear layer : input: 32, output: 1
