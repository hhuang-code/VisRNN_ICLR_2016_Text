import torch
import torch.nn as nn
from torch.autograd import Variable

import sys

from .RNN import *
from .LSTM import *

import pdb

class CharRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, model = "lstm", n_layers = 1):
        super(CharRNN, self).__init__()

        self.model = model.lower()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_layers = n_layers

        self.encoder = nn.Embedding(input_size, input_size)

        if self.model == 'rnn':
            self.rnn = RNN(input_size, hidden_size, n_layers, batch_first = True)
        elif self.model == 'lstm':
            self.rnn = LSTM(input_size, hidden_size, n_layers, batch_first = True)
        elif self.model == 'gru':
            self.rnn = nn.GRU(input_size, hidden_size, n_layers, batch_first = True)
        else:
            raise Exception('No such a model! Exit.')
            sys.exit(-1)

        self.decoder = nn.Linear(hidden_size, output_size)

    def forward(self, input, init_hidden):
        encoded = self.encoder(input)   # input: (batch)
        output, hidden, gates = self.rnn(encoded.view(input.shape[0], 1, -1), init_hidden)   # encoded: (batch, 1, input_size)
        decoded = self.decoder(output) # output: (batch, 1, hidden_size * num_directions)

        return decoded, hidden, gates  # decoded: (batch, seq_len, output_size)

    def init_hidden(self, batch_size):
        if self.model == 'lstm':
            return (Variable(torch.zeros(self.n_layers, batch_size, self.hidden_size)),
                    Variable(torch.zeros(self.n_layers, batch_size, self.hidden_size)))

        return Variable(torch.zeros(self.n_layers, batch_size, self.hidden_size))

