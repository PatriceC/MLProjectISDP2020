import data_preprocessing_seq_to_seq
import LSTM_seq_to_seq

import torch as torch
import torch.nn as nn
import numpy as np

import matplotlib.pyplot as plt

nb_days_before = 7

train, test = data_preprocessing_seq_to_seq.process_data(nb_days_before=nb_days_before, file='./Radar_Traffic_Counts.csv')

data_train = torch.load('./data_seq_train_' + str(nb_days_before) + '.txt')
data_test = torch.load('./data_seq_test_' + str(nb_days_before) + '.txt')

n_train, n_test = data_train.shape[0], data_test.shape[0]
batch_size = 128
data_loader_train, data_loader_test = data_preprocessing_seq_to_seq.data_loader(data_train, data_test, batch_size=batch_size)

model = LSTM_seq_to_seq.LSTM_NN_seq_to_seq(longueur_forecast=24)
error = nn.MSELoss()
learning_rate = 0.0005
weight_decay = 0.0001
num_epoch = 10
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1.0, gamma=0.50)

test_loss_list, pourcentage_loss_list = [], []

count, pourcentage = 0, 0.
for epoch in range(num_epoch):
    for serie, target in data_loader_train:

        output = model.forward(serie)
        target = target.float()
        loss = error(output, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        count += batch_size

        if count >= pourcentage * n_train:
            test_loss_batch = []

            for serie_t, target_t in data_loader_test:

                output_t = model.forward(serie_t)
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

torch.save(model.state_dict(), './LSTM_seq.pt')

#model = LSTM_seq_to_seq.LSTM_NN_seq_to_seq(longueur_forecast=1)
#model.load_state_dict(torch.load('./LSTM_1.pt'))

L = [100, 200, 500, 1000, 2650, 5000, 10000]
for k in L:
    serie, target = torch.tensor([data_test[k][:-24]]), torch.tensor([data_test[k][-24:]])
    pred = model.forward(serie)
    serie = serie.tolist()[0]
    target = target.tolist()[0]
    pred = pred.tolist()[0]

    x, y1, y2 = [x for x in range(len(serie + target))], serie + target, serie + pred
    plt.plot(x, y2, 'r')
    plt.plot(x, y1, 'g')
    plt.show()

"""
plt.figure(0)
plt.plot(pourcentage_loss_list, test_loss_list)
plt.xlabel("Pourcentage * Epochs")
plt.ylabel("MSE Loss")
plt.title("{}: Test Loss vs Pourcentage Epochs".format('LSTM_seq_to_seq'))
plt.show()
"""