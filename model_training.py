# -*- coding: utf-8 -*-
"""
Created on Mon Nov 23 13:54:58 2020

@author: Patrice CHANOL & Corentin MORVAN--CHAUMEIL
"""

import numpy as np
import torch
import time
import visualisation
from datetime import datetime


def main(model, criterion, optimizer, scheduler, data_train_loader, data_test_loader, num_epochs, input_window, output_window, batch_size):
    """
    Entrainement du modèle et Loss Test.

    Parameters
    ----------
    model : TYPE
        DESCRIPTION. model to train
    criterion : TYPE
        DESCRIPTION. criterion to compute
    optimizer : TYPE
        DESCRIPTION.
    scheduler : TYPE
        DESCRIPTION.
    data_loader_train : TYPE
        DESCRIPTION. train set
    data_loader_test : TYPE
        DESCRIPTION. test set
    num_epochs : TYPE
        DESCRIPTION. number of epoch to compute
    input_window : TYPE
        DESCRIPTION. input windonw length
    output_window : TYPE
        DESCRIPTION. output windonw length
    batch_size : TYPE
        DESCRIPTION. batch_size

    Returns
    -------
    model : TYPE
        DESCRIPTION. trained model
    test_loss_list : TYPE
        DESCRIPTION. test loss

    """
    dateTimeObj = datetime.now()
    print('Début Entrainement : ', dateTimeObj.hour, 'H', dateTimeObj.minute)
    test_loss_list = []
    n_batches = len(data_train_loader)
    # On va entrainer le modèle num_epochs fois
    for epoch in range(1, num_epochs + 1):

        # Temps epoch
        epoch_start_time = time.time()
        dateTimeObj = datetime.now()
        print('Début epoch', epoch, ':', dateTimeObj.hour, 'H', dateTimeObj.minute)
        # Modèle en mode entrainement
        model.train()
        # Pourcentage du Dataset réaliser
        pourcentage = 0.
        # Loss du batch en cours
        test_loss_batch = []

        # Temps pour réaliser 10%
        start_time = time.time()

        for batch, ((day_of_week, serie_input), serie_output) in enumerate(data_train_loader):

            # Initializing a gradient as 0 so there is no mixing of gradient among the batches
            optimizer.zero_grad()
            # Forward pass
            output = model.forward(day_of_week, serie_input.float())
            loss = criterion(output, serie_output.float())
            # Propagating the error backward
            loss.backward()

            # Normalisation des gradients si Transformer
            if model.name_model == 'Transformer':
                torch.nn.utils.clip_grad_norm_(model.parameters(), 0.7)

            # Optimizing the parameters
            optimizer.step()

            # Pourcentage réel réaliser
            count_pourcentage = batch / n_batches
            # Si on a réalisé 10% nouveau du Dataset, on test
            if count_pourcentage >= pourcentage:
                # Temps des 10%
                T = time.time() - start_time
                # Evaluation du modèel
                model.eval()
                with torch.no_grad():
                    for ((day_of_week_t, serie_input_t), serie_output_t) in data_test_loader:
                        output_t = model.forward(day_of_week_t, serie_input_t.float())
                        loss_t = criterion(output_t, serie_output_t.float())
                        test_loss_batch.append(loss_t.item())
                test_loss = np.mean(test_loss_batch)
                test_loss_list.append(test_loss)

                print('-'*10)
                print("Pourcentage: {}%, Test Loss : {}, Epoch: {}, Temps : {}s".format(round(100*pourcentage), test_loss, epoch, round(T)))
                print('-'*10)

                # Visualisation
                visualisation.pred_vs_reality(model, input_window, output_window, epoch=epoch, pourcentage=round(100*pourcentage))
                pourcentage += 0.1
                start_time = time.time()

        print('Fin epoch : {}, Temps de l\'epoch : {}s'.format(epoch, round(time.time() - epoch_start_time)))

        visualisation.forecast(model, input_window, output_window, epoch=epoch)

        scheduler.step()

    model.save()

    return model, test_loss_list
