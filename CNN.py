# -*- coding: utf-8 -*-
"""
Created on Sat Nov 28 23:02:46 2020

@author: Patrice CHANOL
"""

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import time

data_train = torch.tensor(np.loadtxt('data_train_12.txt'))
data_test = torch.tensor(np.loadtxt('data_test_12.txt'))
n_train = data_train.shape[0]
n_test = data_test.shape[0]

# Data train format :
# [latitude, longitude, month, day_week, direction] + serie_J + serie_J_moins_1 + serie_J_moins_7 + [target]

latitude_train = data_train[:,0]
latitude_test = data_test[:,0]

longitude_train = data_train[:,1]
longitude_test = data_test[:,1]

month_train = torch.zeros(n_train, 12)
month_test = torch.zeros(n_test, 12)

day_week_train = torch.zeros(n_train, 7)
day_week_test = torch.zeros(n_test, 7)

direction_train = torch.zeros(n_train, 5)
direction_test = torch.zeros(n_test, 5)

for k in range(n_train):
    direction_train[k][int(data_train[k][4])] = 1
    day_week_train[k][int(data_train[k][3])] = 1
    month_train[k][int(data_train[k][2])-1] = 1
    if k < n_test:
        direction_test[k][int(data_test[k][4])] = 1
        day_week_test[k][int(data_test[k][3])] = 1
        month_test[k][int(data_test[k][2])-1] = 1



series_train = data_train[:,5:-1]
series_test = data_test[:,5:-1]

target_train = data_train[:,-1]
target_test = data_test[:,-1]

batch_size = 256

train = torch.utils.data.DataLoader(list(zip(zip(latitude_train, longitude_train, month_train, day_week_train, direction_train, series_train), target_train)), batch_size= batch_size, shuffle=True)
test = torch.utils.data.DataLoader(list(zip(zip(latitude_test, longitude_test, month_test, day_week_test, direction_test, series_test), target_test)), batch_size= batch_size, shuffle=True)

S = 12

# %% Model

class TimeseriesCNN(nn.Module):

    def __init__(self, S):
        super(TimeseriesCNN, self).__init__()

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

        self.layer_tot = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=6, kernel_size=3),
            nn.BatchNorm1d(6),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2)
        )

        self.fc1 = nn.Linear(in_features = int((((3*S-8)/2 * 6)-2)/2 * 6 + 2 + 12 + 7 + 5), out_features=256)
        self.fc2 = nn.Linear(in_features=256, out_features=128)
        self.fc3 = nn.Linear(in_features=128, out_features=64)
        self.fc4 = nn.Linear(in_features=64, out_features=32)
        self.fc5 = nn.Linear(in_features=32, out_features=16)
        self.fc6 = nn.Linear(in_features=16, out_features=1)

    def forward(self, x_1, x_2, x_3, x_latitude, x_longitude, x_month, x_day_week, x_direction):
        out_1 = self.layer_1(x_1)
        out_1 = out_1.view(out_1.size(0), -1)
        out_2 = self.layer_2(x_2)
        out_2 = out_2.view(out_2.size(0), -1)
        out_3 = self.layer_3(x_3)
        out_3 = out_3.view(out_3.size(0), -1)
        out = torch.cat((out_1, out_2, out_3), dim=1)
        out = out.unsqueeze(1)
        out = self.layer_tot(out)
        out = torch.cat((out.view(out.size(0), -1), x_latitude, x_longitude, x_month, x_day_week, x_direction), dim=1)
        out = out.view(out.size(0), -1)
        out = self.fc1(out)
        out = out.relu()
        out = nn.Dropout(0.25)(out)
        out = self.fc2(out)
        out = out.relu()
        out = self.fc3(out)
        out = out.relu()
        out = self.fc4(out)
        out = out.relu()
        out = self.fc5(out)
        out = out.relu()
        out = self.fc6(out)

        return out.view(-1)

# Our input timeserie is changing in a following way:
# Note that S is even
#     Local Convulation layer 1st Serie input : input: (S-1) * 1, output: (S-3) * 6
#     Local Max Pooling layer 1st Serie input: input: (S-3) * 6, output: (S-3)/2 * 6 = (S-4)/2 * 6
#     Local Convulation layer 2nd Serie input: input: S * 1, output: (S-2) * 6
#     Local Max Pooling layer 2nd Serie input: input: (S-2) * 6, output: (S-2)/2 * 6
#     Local Convulation layer 3rd Serie input: input: S * 1, output: (S-2) * 6
#     Local Max Pooling layer 3rd Serie input: input: (S-2) * 6, output: (S-2)/2 * 6
#     Concat of all the Local Conv layers
#     Convolution layer : input : ((3*S-8)/2 * 6) * 1, output: (((3*S-8)/2 * 6)-2) * 6
#     Max Pooling layer : (((3*S-8)/2 * 6)-2) * 6, output: (((3*S-8)/2 * 6)-2)/2 * 6
#     First Linear layer : input: (((3*S-8)/2 * 6)-2)/2 * 6 + 2 + 12 + 7 + 5, output: 256
#     Second Linear layer : input:256, output: 128
#     Third Linear layer : input: 128, output: 64
#     Fourth Linear layer : input: 64, output: 32
#     Fith Linear layer : input: 32, output: 16
#     Sixth Linear layer : input: 16, output: 1

print(TimeseriesCNN(S))

error = nn.MSELoss()

learning_rate = 1

num_epochs = 200
count = 0

loss_list_train = []
loss_list_test = []
loss_test_m = []
loss_train_m = []

model = TimeseriesCNN(S).double()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

t0 = time.time()
for epoch in range(num_epochs):

    loss_te_m = 0
    loss_tr_m = 0

    # Multiplication par 0.1 du Loss toutes 60 epoch à partir de la 30ème
    if not((epoch-30) % 60):
        learning_rate *= 0.1
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    for input_train, target in train:

        # latitude_train, longitude_train, month_train, day_week_train, direction_train, series_train
        latitude = input_train[0].unsqueeze(1)
        longitude = input_train[1].unsqueeze(1)
        month = input_train[2]
        day_week = input_train[3]
        direction = input_train[4]
        serie_1 = input_train[5][:,:11].unsqueeze(1)
        serie_2 = input_train[5][:,11:23].unsqueeze(1)
        serie_3 = input_train[5][:,23:].unsqueeze(1)
    
        # Forward pass
        outputs = model(serie_1.double(), serie_2.double(), serie_3.double(), latitude, longitude, month, day_week, direction)

        loss = error(outputs, target)

        # Initializing a gradient as 0 so there is no mixing of gradient among the batches
        optimizer.zero_grad()

        # Propagating the error backward
        loss.backward()

        # Optimizing the parameters
        optimizer.step()

        loss_tr_m += loss.item()
        loss_list_train.append(loss.item())
    loss_train_m.append(loss_tr_m/len(train))

    # Testing the model
    for input_test, target in test:

        # latitude_train, longitude_train, month_train, day_week_train, direction_train, series_train
        latitude = input_test[0].unsqueeze(1)
        longitude = input_test[1].unsqueeze(1)
        month = input_test[2]
        day_week = input_test[3]
        direction = input_test[4]
        serie_1 = input_test[5][:,:11].unsqueeze(1)
        serie_2 = input_test[5][:,11:23].unsqueeze(1)
        serie_3 = input_test[5][:,23:].unsqueeze(1)
        
        outputs = model(serie_1.double(), serie_2.double(), serie_3.double(), latitude, longitude, month, day_week, direction)

        loss = error(outputs, target)

        loss_te_m += loss.item()
        loss_list_test.append(loss.item())
    loss_test_m.append(loss_te_m/len(test))
    print("Iteration: {}, Loss train: {}, Loss test: {}".format(count, loss_train_m[-1], loss_test_m[-1]))
    count += 1

t1 = time.time()
total = t0 - t1

# Plot
print(total)
plt.plot(loss_list_test)
plt.xlabel("epoch")
plt.ylabel("Loss")
plt.title("Test")
plt.show()
plt.plot(loss_test_m)
plt.xlabel("epoch * nb_batch")
plt.ylabel("Loss")
plt.title("Test mean")
plt.show()
plt.plot(loss_list_train)
plt.xlabel("epoch * nb_batch")
plt.ylabel("Loss")
plt.title("Train")
plt.show()
plt.plot(loss_train_m)
plt.xlabel("epoch")
plt.ylabel("Loss")
plt.title("Train mean")
plt.show()
