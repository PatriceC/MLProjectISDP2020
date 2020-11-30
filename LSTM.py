import torch as torch
import torch.nn as nn

class LSTM_NN(nn.Module):
    """

    """
    def __init__(self, longueur_serie):
        super(LSTM_NN, self).__init__()

        self.lstm_1 = nn.LSTM(input_size=longueur_serie, hidden_size=100, num_layers=1, batch_first=True, dropout=0.2)
        self.lstm_2 = nn.LSTM(input_size=longueur_serie + 100, hidden_size=100, num_layers=1, batch_first=True, dropout=0.2)
        self.lstm_3 = nn.LSTM(input_size=longueur_serie - 1 + 100, hidden_size=100, num_layers=1, batch_first=True, dropout=0.2)

        self.fc_1 = nn.Linear(in_features=100 + 2 + 12 + 7 + 5, out_features=32)
        self.fc_2 = nn.Linear(in_features=32, out_features=8)
        self.fc_3 = nn.Linear(in_features=8, out_features=1)
        self.relu = nn.ReLU()

    
    def forward(self, latitude, longitude, month, day_week, direction, serie_J, serie_J_moins_1, serie_J_moins_7):

        out, _  = self.lstm_1(serie_J_moins_1)
        out, _  = self.lstm_2(torch.cat((out,serie_J_moins_7),dim=2))
        out, _  = self.lstm_3(torch.cat((out,serie_J),dim=2))

        out = torch.cat((out, latitude.unsqueeze(1), longitude.unsqueeze(1), month.float().unsqueeze(1), day_week.float().unsqueeze(1), direction.float().unsqueeze(1)), dim=2)
        out = self.fc_1(out)
        out = self.relu(out)
        out = self.fc_2(out)
        out = self.relu(out)
        out = self.fc_3(out)
        out = self.relu(out)
        return(out)