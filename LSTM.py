import torch as torch
import torch.nn as nn

class LSTM_NN(nn.Module):
    """

    """
    def __init__(self):
        super(LSTM_NN, self).__init__()

        self.lstm_layer = nn.LSTM(input_size=5, hidden_size=100, num_layers=1, batch_first=True)
        self.dropout = nn.Dropout(0.2)
        self.fc = nn.Linear(in_features=100 + 2 + 12 + 7 + 5, out_features=1)
        self.relu = nn.ReLU()

    
    def forward(self, latitude, longitude, month, day_week, direction, serie_J, serie_J_moins_1, serie_J_moins_7):

        out, _ = self.lstm_layer(serie_J)
        out = self.dropout(out)
        out = torch.cat((out, latitude.unsqueeze(1), longitude.unsqueeze(1), month.float().unsqueeze(1), day_week.float().unsqueeze(1), direction.float().unsqueeze(1)), dim=2)
        out = self.fc(out)
        out = self.relu(out)
        return(out)