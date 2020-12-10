# -*- coding: utf-8 -*-
"""
Created on Sat Dec  5 23:49:32 2020

@author: Patrice CHANOL
"""
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import time
import math
import matplotlib.pyplot as plt

longueur_serie = 23
batch_size = 256
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# %% Model

class PositionalEncoding(nn.Module):

    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:x.size(0), :]


class Transformer(nn.Module):
    def __init__(self, feature_size=230, num_layers=1, dropout=0.1, longueur_serie=23):
        super(Transformer, self).__init__()
        self.model_type = 'Transformer'

        self.src_mask = None
        self.pos_encoder = PositionalEncoding(feature_size)
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=feature_size, nhead=23, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers).float()
        self.decoder = nn.Linear(feature_size, 1)
        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, src):
        if self.src_mask is None or self.src_mask.size(0) != len(src):
            device = src.device
            mask = self._generate_square_subsequent_mask(len(src)).to(device)
            self.src_mask = mask

        src = self.pos_encoder(src)
        output = self.transformer_encoder(src, self.src_mask)
        output = self.decoder(output)
        return output

    def _generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

# %% Data Preprocessing


def series(Date_J, latitude, longitude, direction, longueur_serie, data):
    """
    Retourne 1 séries de longueur_serie valeurs de Volume pour un jour.

    Parameters
    ----------
    Date_J : TYPE datetime
        DESCRIPTION. Date des séries à prédire
    latitude : TYPE float
        DESCRIPTION. Latitude des séries
    longitude : TYPE float
        DESCRIPTION. Longitude des séries
    direction : TYPE int
        DESCRIPTION. Direction des séries
    longueur_serie : TYPE int
        DESCRIPTION. Longueur des séries
    data : TYPE DataFrame
        DESCRIPTION. Donnée
    """
    # Récupération des données de Date_J (au jour J, J-1)
    row_J = data[(data['location_latitude'] == latitude) & (data['location_longitude'] == longitude) & (data['Date'] == Date_J) & (data['Direction'] == direction)]
    row_J_moins_1 = data[(data['location_latitude'] == latitude) & (data['location_longitude'] == longitude) & (data['Date'] == Date_J - pd.to_timedelta(1, unit='d')) & (data['Direction'] == direction)]

    # Si on a pas de donnée pour le jour J
    if row_J.empty:
        return None

    # Sinon si on a pas de données pour le jour J-1, on renvoit juste les séries disponibles dans la journée
    elif row_J_moins_1.empty:
        valeurs_J = row_J.values.tolist()[0][8:]

        nb_series = 24 - longueur_serie
        target, serie_J = np.zeros((nb_series, longueur_serie)), np.zeros((nb_series, longueur_serie))

        for h in range(nb_series):
            target[h] = valeurs_J[h + 1 : h + 1 +longueur_serie]
            serie_J[h] = valeurs_J[h : h + longueur_serie]

    # Sinon on dispose de toute les données et on peut générer 24 séries
    else:
        valeurs_J, valeurs_J_moins_1 = row_J.values.tolist()[0][8:], row_J_moins_1.values.tolist()[0][8:]
        V_J = valeurs_J_moins_1 + valeurs_J

        nb_series = 24
        target, serie_J = np.zeros((nb_series, longueur_serie)), np.zeros((nb_series, longueur_serie))

        for h in range(nb_series):
            target[h] = V_J[h + nb_series - longueur_serie + 1 : h + nb_series + 1]
            serie_J[h] = V_J[h + nb_series - longueur_serie : h + nb_series]

    return(target, serie_J)


def process_data(longueur_serie=23, file='./Radar_Traffic_Counts.csv'):
    """
    Génération du Dataset désiré.

    Parameters
    ----------
    date_range : TYPE, optional
        DESCRIPTION. The default is ['2017', '2020'].
    direction : TYPE, optional
        DESCRIPTION. The default is None.
    latitude : TYPE, optional
        DESCRIPTION. The default is [-100, 100].
    longitude : TYPE, optional
        DESCRIPTION. The default is [-100, 100].
    longueur_serie : TYPE, optional
        DESCRIPTION. The default is 6.
    file : TYPE, optional
        DESCRIPTION. The default is './Radar_Traffic_Counts.csv'.

    """
    data = pd.read_csv(file)

    # Préparation du Dataset
    data = data.drop(columns=['location_name', 'Time Bin'])
    data['Direction'] = data['Direction'].astype('category').cat.codes
    data['Date'] = pd.to_datetime(data[['Year', 'Month', 'Day']], errors='coerce')
    data = data.sort_values(['Date', 'Hour'])

    data['location_latitude'] = data['location_latitude']
    data['location_longitude'] = data['location_longitude']
    col = ['location_latitude', 'location_longitude', 'Year', 'Month', 'Day', 'Date', 'Day of Week', 'Hour', 'Direction']
    col_no_hour = ['location_latitude', 'location_longitude', 'Year', 'Month', 'Day', 'Date', 'Day of Week', 'Direction']
    data = data.groupby(col)['Volume'].sum().reset_index()
    # On va normaliser (méthode min-max) les valeurs
    volume_max, volume_min = data['Volume'].max(), data['Volume'].min()
    data = data.pivot_table(index=col_no_hour, columns='Hour', values='Volume').reset_index()
    # Suppression des jours contenant des données manquantes
    data = data.dropna()
    # On garde les valeurs de mois entre 0 et 11 (plutôt que 1 et 12), ce qui sera plus pratique pour créer des one-hot vectors
    data['Month'] = data['Month'] - 1

    data_train, data_test = [], []
    # Pour chaque jour
    k = 0
    for _, row in data.iterrows():
        k += 1
        # On récuprère les informations de cette donnée
        latitude, longitude = row['location_latitude'], row['location_longitude']
        date, direction = row['Date'], row['Direction']
        # On génère les séries
        result = series(Date_J=date, latitude=latitude, longitude=longitude, direction=direction, longueur_serie=longueur_serie, data=data)

        if result is not None:
            target, serie_J = result
            # On récupère les heures pour plot
            for t, s1 in zip(target, serie_J):
                # On normalise les valeurs
                s1_norm = list((s1 - volume_min)/(volume_max - volume_min))
                t_norm = list((t - volume_min)/(volume_max - volume_min))
                # On sépare le dataset en 90% training et 10% test
                if k < 0.9*len(data):
                    data_train.append([s1_norm, t_norm])
                else:
                    data_test.append([s1_norm, t_norm])
    # Enregistrement des données

    data_train = torch.tensor(data_train).float()
    data_test = torch.tensor(data_test).float()
    torch.save(data_train, 'data_train_Att_' + str(longueur_serie) + '.txt')
    torch.save(data_test, 'data_test_Att_' + str(longueur_serie) + '.txt')

    return(data_train, data_test, volume_max, volume_min)

# %% Forecast


def forecast(epoch=0, steps=100):
    """
    Forecast du Transformer.

    Parameters
    ----------
    epoch : TYPE, optional
        DESCRIPTION. The default is 0.
    steps : TYPE, optional
        DESCRIPTION. The default is 100.

    Returns
    -------
    None.

    """
    model.eval()
    data = data_test[0, 0].unsqueeze(1).unsqueeze(2)
    with torch.no_grad():
        for i in range(0, steps):
            output = model(data[-longueur_serie:])
            data = torch.cat((data, output[-1:]))

    data = data.cpu().view(-1)
    truth = data_test[1:len(data), 1, -1]

    plt.figure(10000 + epoch)
    plt.plot(truth, color="green")
    plt.plot(data, color="red")
    plt.plot(data[:longueur_serie], color="blue")
    plt.axis([0, len(data), 0, 0.35])
    plt.savefig('attention_graph/transformer-future-{}-{}.png'.format(epoch, steps), dpi=300)
    plt.figure(20000 + epoch)
    plt.plot(data, color="red")
    plt.plot(data[:longueur_serie], color="blue")
    plt.axis([0, len(data), 0, 0.35])
    plt.savefig('attention_graph/transformer-future2-{}-{}.png'.format(epoch, steps), dpi=300)

# %% Data

# data_train, data_test, volume_max, volume_min = process_data(longueur_serie)
data_train = torch.load('data_train_Att_23.txt')
data_test = torch.load('data_test_Att_23.txt')
n_train, n_test = len(data_train), len(data_test)

data_train_loader = torch.utils.data.DataLoader(list(zip(data_train[:, 0, :], data_train[:, 1, :])), batch_size=batch_size, shuffle=True)
data_test_loader = torch.utils.data.DataLoader(list(zip(data_test[:, 0, :], data_test[:, 1, :])), batch_size=2*batch_size)

# %% Model Training & Testing

model = Transformer(230).to(device)

error = nn.MSELoss()
lr = 0.005
optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1.0, gamma=0.90)

num_epochs = 20

test_loss_list = []
for epoch in range(1, num_epochs + 1):
    epoch_start_time = time.time()
    model.train()
    count, pourcentage = 0, 0.
    train_loss_batch, test_loss_batch = [], []
    pourcentage_loss_list = []
    start_time = time.time()

    for data, targets in data_train_loader:
        data = data.transpose(0, 1).unsqueeze(2)
        targets = targets.transpose(0, 1).unsqueeze(2)
        optimizer.zero_grad()
        output = model(data)
        loss = error(output, targets)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.7)
        optimizer.step()
        train_loss_batch.append(loss.item())

        count += batch_size
        if count >= pourcentage * n_train:
            plt.close('all')
            T = time.time() - start_time

            model.eval()
            test_result = torch.Tensor(0)
            truth = torch.Tensor(0)
            with torch.no_grad():
                for data_t, targets_t in data_test_loader:
                    data_t = data_t.transpose(0, 1).unsqueeze(2)
                    targets_t = targets_t.transpose(0, 1).unsqueeze(2)
                    output_t = model(data_t)
                    test_loss_batch.append(error(output_t, targets_t).item())
                    test_result = torch.cat((test_result, output_t[-1].view(-1).cpu()), 0)
                    truth = torch.cat((truth, targets_t[-1].view(-1).cpu()), 0)
            test_loss = np.mean(test_loss_batch)
            test_loss_list.append(test_loss)

            print("Pourcentage: {}%, Test Loss : {}, Epoch: {}, Time : {}".format(int(round(100*pourcentage)), test_loss, epoch, T))
            print()

            plt.figure(epoch*int(round(100*pourcentage)))
            plt.plot(truth[10000:10500], color="blue", label='Data')
            plt.plot(test_result[10000:10500], color="red", label='Pred')
            plt.title('Data vs Pred')
            plt.axis([0, 500, 0, 0.35])
            plt.legend(loc='upper center')
            plt.show()
            plt.savefig('attention_graph/transformer-epoch{}-{}%.png'.format(epoch, int(round(100*pourcentage))), dpi=300)

            pourcentage_loss_list.append(int(round(100*pourcentage)))
            pourcentage += 0.05
            start_time = time.time()

    print('Time for Epoch : {:5.2f}s '.format(time.time() - epoch_start_time))

    forecast(epoch)

    scheduler.step()

torch.save(model, 'TransformerV3.mod')