import torch as torch
import torch.nn as nn

class LSTM_NN(nn.Module):
    """

    """
    def __init__(self, longueur_serie):
        super(LSTM_NN, self).__init__()

        self.lstm = nn.LSTM(input_size=3, hidden_size=100, num_layers=1, batch_first=True)

        self.dropout = nn.Dropout(0.1)

        self.lin = nn.Linear(in_features=100 + 1 + 1 + 1 + 12 + 7 + 5, out_features=32)
        self.lin2 = nn.Linear(in_features=32, out_features=1)
        self.relu = nn.ReLU()

    def forward(self, latitude, longitude, month, day_week, direction, serie_J, serie_J_moins_1, serie_J_moins_7):

        latitude = latitude.float().unsqueeze(1)
        longitude = longitude.float().unsqueeze(1)
        month = month.float()
        day_week = day_week.float()
        direction = direction.float()
        serie_J = serie_J.float().unsqueeze(2)
        serie_J_moins_1 = serie_J_moins_1.float().unsqueeze(2)
        serie_J_moins_7 = serie_J_moins_7.float().unsqueeze(2)
    
        input = torch.cat((serie_J_moins_1[:,:-1,:], serie_J_moins_7[:,:-1,:], serie_J), dim=2)

        out, _ = self.lstm(input)

        out = out[:,-1,:]

        out = self.dropout(out)

        input_2 = torch.cat((out, latitude, longitude, month, day_week, direction, serie_J_moins_7[:,-1,:]), dim=1)

        out = self.lin(input_2)
        out = self.relu(out)
        out = self.lin2(out)
        return(out.squeeze(1))