# -*- coding: utf-8 -*-
"""
Created on Mon Nov 23 13:54:58 2020

@author: Patrice CHANOL & Corentin MORVAN--CHAUMEIL
"""

import matplotlib.pyplot as plt
import numpy as np
import torch


def main(nom_model, model, error, data_loader_train, data_loader_test, n_train, optimizer, scheduler, num_epoch, batch_size):
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
        for (latitude, longitude, month, day_week, direction, serie_J, serie_J_moins_1, serie_J_moins_7), target in data_loader_train:

            output = model.forward(latitude, longitude, month, day_week, direction, serie_J, serie_J_moins_1, serie_J_moins_7)
            loss = error(output, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            count += batch_size

            if count >= pourcentage * n_train:
                test_loss_batch = []

                for (latitude_t, longitude_t, month_t, day_week_t, direction_t, serie_J_t, serie_J_moins_1_t, serie_J_moins_7_t), target_t in data_loader_test:

                    output_t = model.forward(latitude_t, longitude_t, month_t, day_week_t, direction_t, serie_J_t, serie_J_moins_1_t, serie_J_moins_7_t)
                    loss_test = error(output_t, target_t).data.item()
                    test_loss_batch.append(loss_test)

                test_loss = np.mean(test_loss_batch)
                print("Pourcentage * Epoch: {}%, Epoch: {}".format(int(round(100*pourcentage)), epoch+1))
                print(test_loss)
                print()
                pourcentage_loss_list.append(int(round(100*pourcentage)))
                test_loss_list.append(test_loss)
                pourcentage += 0.1
        scheduler.step()
    plt.figure(0)
    plt.plot(pourcentage_loss_list, test_loss_list)
    plt.xlabel("Pourcentage * Epochs")
    plt.ylabel("MSE Loss")
    plt.title("{}: Test Loss vs Pourcentage Epochs".format(nom_model))
    plt.show()

    return model, pourcentage_loss_list, test_loss_list