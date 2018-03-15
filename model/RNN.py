import torch
import torch.nn as nn
import torch.nn.functional as F

import pdb

class RNN(nn.Module):

    def __init__(self, input_size, hidden_size, num_layers = 1, batch_first = True):
        super(RNN, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.batch_first = batch_first

        # note that nn.Linear contains bias
        self.Wih_b = nn.Linear(input_size, hidden_size, bias = True)    # input weight
        self.Whh_b = nn.Linear(hidden_size, hidden_size, bias = True)   # hidden weight

    def forward(self, input, hidden):

        # recurrence helper, function for one time step
        def recurrence(input, hidden):
            # previous state
            hx = hidden # hx: (num_layers, batch_size, hidden_size)
            # current state
            hy = F.tanh(self.Wih_b(input) + self.Whh_b(hx))

            return hy

        if self.batch_first:
            input = input.transpose(0, 1)

        output = [] # a list, call cat() before returned
        steps = range(input.size(0))    # seq_len

        for i in steps: # process for each tiem step
            hidden = recurrence(input[i], hidden)
            if isinstance(hidden, tuple):
                output.append(hidden[0])
            else:
                output.append(hidden)

        output = torch.cat(output, 0)

        if self.batch_first:
            output = output.transpose(0, 1)

        return output, hidden