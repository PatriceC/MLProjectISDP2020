# -*- coding: utf-8 -*-
"""
Created on Mon Nov 23 13:54:58 2020

@author: Patrice CHANOL
"""

import matplotlib.pyplot as plt
import numpy as np
import torch


def main(model, error, data_loader_train, data_loader_test, n_train, learning_rate, lr_dim, weight_decay, num_epoch, batch_size):
    """
    
    Entrainement du modèle et Loss Test

    Parameters
    ----------
    model : TYPE
        DESCRIPTION. model to train
    error : TYPE
        DESCRIPTION. error to compute
    data_loader_train : TYPE
        DESCRIPTION. train set
    data_loader_test : TYPE
        DESCRIPTION. test set
    n_train : TYPE
        DESCRIPTION. train dataset length
    learning_rate : TYPE
        DESCRIPTION. model learning rate
    lr_dim : TYPE
        DESCRIPTION. diminution of learning rate
    weight_decay : TYPE
        DESCRIPTION. weight decay
    num_epoch : TYPE
        DESCRIPTION. number of epoch to compute
    batch_size : TYPE
        DESCRIPTION. batch_size

    Returns
    -------
    model : TYPE
        DESCRIPTION. trained model
    error : TYPE
        DESCRIPTION. model error used
    pourcentage_loss_list : TYPE
        DESCRIPTION. percentage done
    test_loss_list : TYPE
        DESCRIPTION. test loss

    """

    test_loss_list, pourcentage_loss_list = [], []

    count, pourcentage = 0, 0.
    for epoch in range(num_epoch):
        learning_rate /= lr_dim
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        for (latitude, longitude, month, day_week, direction, serie_J, serie_J_moins_1, serie_J_moins_7), target in data_loader_train:

            latitude = latitude.float().unsqueeze(1)
            longitude = longitude.float().unsqueeze(1)
            month = month.float()
            day_week = day_week.float()
            direction = direction.float()
            serie_J = serie_J.float().unsqueeze(1)
            serie_J_moins_1 = serie_J_moins_1.float().unsqueeze(1)
            serie_J_moins_7 = serie_J_moins_7.float().unsqueeze(1)
            target = target.float().unsqueeze(1)

            output = model.forward(latitude, longitude, month, day_week, direction, serie_J, serie_J_moins_1, serie_J_moins_7)
            loss = error(output, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            count += batch_size

            if count >= pourcentage * n_train:
                test_loss_batch = []

                for (latitude_t, longitude_t, month_t, day_week_t, direction_t, serie_J_t, serie_J_moins_1_t, serie_J_moins_7_t), target_t in data_loader_test:

                    latitude_t = latitude_t.float().unsqueeze(1)
                    longitude_t = longitude_t.float().unsqueeze(1)
                    month = month.float()
                    day_week = day_week.float()
                    direction = direction.float()
                    serie_J_t = serie_J_t.float().unsqueeze(1)
                    serie_J_moins_1_t = serie_J_moins_1_t.float().unsqueeze(1)
                    serie_J_moins_7_t = serie_J_moins_7_t.float().unsqueeze(1)
                    target_t = target_t.float().unsqueeze(1)

                    output_t = model.forward(latitude_t, longitude_t, month_t, day_week_t, direction_t, serie_J_t, serie_J_moins_1_t, serie_J_moins_7_t)
                    loss_test = error(output_t, target_t).data.item()
                    test_loss_batch.append(loss_test)

                test_loss = np.mean(test_loss_batch)
                print("Pourcentage * Epoch: {}%, Epoch: {}".format(100*pourcentage, epoch+1))
                print(test_loss)
                print()
                pourcentage_loss_list.append(100*pourcentage)
                test_loss_list.append(test_loss)
                pourcentage += 0.1

    plt.plot(pourcentage_loss_list, test_loss_list)
    plt.xlabel("Pourcentage * Epochs")
    plt.ylabel("MAE Loss")
    plt.title("Test Loss vs Pourcentage Epochs")
    plt.show()

    return model, error, pourcentage_loss_list, test_loss_list