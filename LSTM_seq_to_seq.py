import torch as torch
import torch.nn as nn

class LSTM_NN_seq_to_seq(nn.Module):
    """

    """
    def __init__(self, longueur_forecast):
        super(LSTM_NN_seq_to_seq, self).__init__()

        self.lstm = nn.LSTM(input_size=1, hidden_size=200, num_layers=1, batch_first=True)

        self.dropout = nn.Dropout(0.1)

        self.lin = nn.Linear(in_features=200, out_features=72)
        self.lin2 = nn.Linear(in_features=72, out_features=longueur_forecast)
        self.relu = nn.ReLU()

    def forward(self, serie):

        serie = serie.float().unsqueeze(2)

        out, _ = self.lstm(serie)

        out = out[:, -1, :]

        out = self.dropout(out)

        out = self.lin(out)
        out = self.relu(out)
        out = self.lin2(out)
        return(out)