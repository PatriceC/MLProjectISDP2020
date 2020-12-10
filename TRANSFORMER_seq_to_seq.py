# -*- coding: utf-8 -*-
"""
Created on Sat Dec  5 23:49:32 2020

@author: Patrice CHANOL & Corentin MORVAN-CHAUMEIL

https://github.com/oliverguhr/transformer-time-series-prediction/blob/master/transformer-multistep.py
"""
import torch
import torch.nn as nn
import math

# %% Model


class PositionalEncoding(nn.Module):
    """Positional Encoder."""

    def __init__(self, d_model, max_len=5000):
        """Positional Encoder (see Attention is all you need)."""
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """Forward Pass."""
        return x + self.pe[:x.size(0), :]


class Transformer(nn.Module):
    """Implémentation d'un Transformer seq-to-seq."""

    def __init__(self, input_window, output_window, num_layers=1, dropout=0.1, longueur_serie=23):
        """
        Init.

        Parameters
        ----------
        input_window: int
            Représente le nombre de jour de la séquence d'entrée
            Longueur de la séquence d'entrée: 24 * input_window
        output_window: int
            Représente le nombre d'heure de la séquence de sortie
            Longueur de la séquence de sortie: output_window
        """
        super(Transformer, self).__init__()

        self.name_model = 'Transformer'
        self.input_window = input_window
        self.output_window = output_window
        self.feature_size = self.output_window * 4

        self.src_mask = None
        self.pos_encoder = PositionalEncoding(self.feature_size)
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=self.feature_size, dim_feedforward=self.feature_size * 4, nhead=self.output_window, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers).float()
        self.decoder = nn.Linear(self.feature_size, 1)
        self.init_weights()

    def init_weights(self):
        """Init weights."""
        initrange = 0.1
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, day_of_week, src):
        """Forward Pass."""
        src = torch.cat((src, torch.zeros(src.size(0), self.output_window)), dim=1)
        src = src.transpose(0, 1).unsqueeze(2)
        if self.src_mask is None or self.src_mask.size(0) != len(src):
            device = src.device
            mask = self._generate_square_subsequent_mask(len(src)).to(device)
            self.src_mask = mask

        src = self.pos_encoder(src)
        output = self.transformer_encoder(src, self.src_mask)
        output = self.decoder(output)
        output = output.squeeze(2).transpose(0, 1)
        output = output[:, -self.output_window:]
        return output

    def _generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def save(self):
        """Enregistre le modèle pour inférence dans le futur."""
        torch.save(self.state_dict(), './models/model_' + self.name_model + '_' + str(self.input_window) + '_days_to_' + str(self.output_window) + '_hours.pt')

    def load(self):
        """Récupère un modèle déjà entrainé pour inférer."""
        self.load_state_dict(torch.load('./models/model_' + self.name_model + '_' + str(self.input_window) + '_days_to_' + str(self.output_window) + '_hours.pt'))
