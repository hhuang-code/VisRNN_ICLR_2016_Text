import torch
import torch.nn as nn
from torch.autograd import Variable

import sys

import pdb

class CharRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, model = "lstm", n_layers = 1):
        super(CharRNN, self).__init__()

        self.model = model.lower()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_layers = n_layers

        self.embed = nn.Embedding(input_size, hidden_size)

        if self.model == "rnn":
            self.rnn = nn.RNN(hidden_size, hidden_size, n_layers, batch_first = True)
        elif self.model == "lstm":
            self.rnn = nn.LSTM(hidden_size, hidden_size, n_layers, batch_first = True)
        elif self.model == "gru":
            self.rnn = nn.GRU(hidden_size, hidden_size, n_layers, batch_first = True)
        else:
            raise Exception('No such a model! Exit.')
            sys.exit(-1)

        self.proj = nn.Linear(hidden_size, output_size)

    def forward(self, input, init_hidden):
        encoded = self.embed(input) # encoded: (batch, seq_length, input_size)
        output, hidden = self.rnn(encoded, init_hidden)
        decoded = self.proj(output) # output: (batch, seq_length, hidden_size * num_directions)

        return decoded, hidden

    def init_hidden(self, batch_size):
        # if self.model == "lstm":
        #     return (Variable(torch.Tensor(self.n_layers, batch_size, self.hidden_size).uniform_(-0.08, 0.08)),
        #             Variable(torch.Tensor(self.n_layers, batch_size, self.hidden_size).uniform_(-0.08, 0.08)))
        #
        # return Variable(torch.Tensor(self.n_layers, batch_size, self.hidden_size).uniform_(-0.08, 0.08))

        if self.model == "lstm":
            return (Variable(torch.zeros(self.n_layers, batch_size, self.hidden_size)),
                    Variable(torch.zeros(self.n_layers, batch_size, self.hidden_size)))

        return Variable(torch.zeros(self.n_layers, batch_size, self.hidden_size))

