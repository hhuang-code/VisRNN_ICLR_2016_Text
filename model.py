# https://github.com/spro/char-rnn.pytorch

import torch
import torch.nn as nn
from torch.autograd import Variable

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

        if self.model == "gru":
            self.rnn = nn.GRU(hidden_size, hidden_size, n_layers, batch_first = True)
        elif self.model == "lstm":
            self.rnn = nn.LSTM(hidden_size, hidden_size, n_layers, batch_first = True)

        self.proj = nn.Linear(hidden_size, output_size)

    def forward(self, input, init_hidden):
        encoded = self.embed(input) # encoded: (batch, seq_length, input_size)
        output, (h_n, c_n) = self.rnn(encoded, init_hidden)
        decoded = self.proj(output) # output: (batch, seq_length, hidden_size * num_directions)

        return decoded, (h_n, c_n)

    # def forward2(self, input, hidden):
    #     encoded = self.encoder(input.view(1, -1))
    #     output, hidden = self.rnn(encoded.view(1, 1, -1), hidden)
    #     output = self.decoder(output.view(1, -1))
    #     return output, hidden

    def init_hidden(self, batch_size):
        if self.model == "lstm":
            return (Variable(torch.zeros(self.n_layers, batch_size, self.hidden_size)),
                    Variable(torch.zeros(self.n_layers, batch_size, self.hidden_size)))

        return Variable(torch.zeros(self.n_layers, batch_size, self.hidden_size))

