import torch as torch
import torch.nn as nn

class LSTM_NN(nn.Module):
    """

    """
    def __init__(self):
        super(LSTM_NN, self).__init__()

        self.lstm_layer = nn.LSTM(input_size=5, hidden_size=1, num_layers=1, batch_first=True)
    
    def forward(self, x):

        out = self.lstm_layer(x)
        return(out)

a = torch.tensor([[[1.,2.,3.,4.,5.]]])
model = LSTM_NN()
b, (c, d) = model.forward(a)
print(b)
print(c, d)