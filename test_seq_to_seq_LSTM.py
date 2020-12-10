import data_preprocessing
import LSTM_seq_to_seq
import CNN_seq_to_seq
import visualisation

import torch as torch
import torch.nn as nn
import numpy as np

import matplotlib.pyplot as plt

input_window, output_window = 1, 1
#train, test = data_preprocessing.process_data(input_window=input_window, output_window=output_window, file='./Radar_Traffic_Counts.csv')

data_train = torch.load('./data/data_train_' + str(input_window) + '_days_to_' + str(output_window) + '_hours.txt')
data_test = torch.load('./data/data_test_' + str(input_window) + '_days_to_' + str(output_window) + '_hours.txt')

n_train, n_test = data_train.shape[0], data_test.shape[0]
batch_size = 128
data_loader_train, data_loader_test = data_preprocessing.data_loader(data_train, data_test, input_window, output_window, batch_size=batch_size)

model = LSTM_seq_to_seq.LSTM(input_window=input_window, output_window=output_window)
error = nn.MSELoss()
learning_rate = 0.001
weight_decay = 0.0001
num_epoch = 1
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1.0, gamma=0.50)

test_loss_list, pourcentage_loss_list = [], []

count, pourcentage = 0, 0.
for epoch in range(num_epoch):
    for (day_of_week, serie), target in data_loader_train:

        output = model.forward(day_of_week, serie)
        target = target.float()
        loss = error(output, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        count += batch_size

        if count >= pourcentage * n_train:
            test_loss_batch = []

            for (day_of_week_t, serie_t), target_t in data_loader_test:

                output_t = model.forward(day_of_week_t, serie_t)
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

visualisation.forecast(model,1,1)
visualisation.pred_vs_reality(model, 1, 1)
"""
L = [100, 200, 500, 1000, 2650, 5000, 10000]
for k in L:
    day_week, serie, target = torch.tensor([data_test[k][:7]]), torch.tensor([data_test[k][7:-24]]), torch.tensor([data_test[k][-24:]])
    pred = model.forward(day_week, serie)
    serie = serie.tolist()[0]
    target = target.tolist()[0]
    pred = pred.tolist()[0]

    x, y1, y2 = [x for x in range(len(serie + target))], serie + target, serie + pred
    plt.plot(x, y2, 'r', label="Prédiction")
    plt.plot(x, y1, 'g', label="Réalité")
    plt.title("Data vs Pred")
    plt.show()
"""
"""
plt.figure(0)
plt.plot(pourcentage_loss_list, test_loss_list)
plt.xlabel("Pourcentage * Epochs")
plt.ylabel("MSE Loss")
plt.title("{}: Test Loss vs Pourcentage Epochs".format('LSTM_seq_to_seq'))
plt.show()
"""