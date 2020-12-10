# -*- coding: utf-8 -*-
"""
Created on Mon Nov 23 13:54:58 2020

@author: Patrice CHANOL & Corentin MORVAN--CHAUMEIL
"""

import numpy as np
import torch
import time
import visualisation

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
    test_loss_list = []
    n_batches = len(data_train_loader)
    for epoch in range(1, num_epochs + 1):
        epoch_start_time = time.time()
        model.train()
        pourcentage = 0.
        test_loss_batch = []
        start_time = time.time()
    
        for batch, (day_of_week, serie_input, serie_output) in enumerate(data_train_loader):
            optimizer.zero_grad()
            output = model.forward(day_of_week, serie_input)

            if model.name_model == 'Transformer':
                serie_output = serie_output
                loss = criterion(output, serie_output)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 0.7)
            else:
                loss = criterion(output, serie_output)
                loss.backward()
                
            optimizer.step()
    
            count_pourcentage = round(100*(batch / n_batches))
            if count_pourcentage >= pourcentage:
                T = time.time() - start_time
                model.eval()
                with torch.no_grad():
                    for (day_of_week_t, serie_input_t, serie_output_t) in data_test_loader:
                        output_t = model.forward(day_of_week_t, serie_input_t)
                        if model.name_model == 'Transformer':
                            serie_output_t = serie_output_t
                            loss_t = criterion(output_t, serie_output_t)
                        else:
                            loss_t = criterion(output_t, serie_output_t)

                        test_loss_batch.append(loss_t.item())
                test_loss = np.mean(test_loss_batch)
                test_loss_list.append(test_loss)
    
                print('-'*10)
                print("Pourcentage: {}%, Test Loss : {}, Epoch: {}, Temps : {}s".format(round(100*pourcentage), test_loss, epoch, round(T)))
                print('-'*10)
                visualisation.pred_vs_reality(model, input_window, output_window)
                pourcentage += 0.1
                start_time = time.time()
    
        print('Fin epoch : {}, Temps de l\'epoch : {}s'.format(epoch, round(time.time() - epoch_start_time)))
    
        visualisation.forecast(model, input_window, output_window)
    
        scheduler.step()
    return model, test_loss_list