# Thanks to https://discuss.pytorch.org/t/implementation-of-multiplicative-lstm/2328/9

import torch
import torch.nn as nn
import torch.nn.functional as F

import pdb

class SLSTM(nn.Module):

    def __init__(self, input_size, hidden_size, num_layers = 1, batch_first = True):
        super(SLSTM, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.batch_first = batch_first

        # note that nn.Linear contains bias
        self.Wii_b = nn.Linear(input_size, hidden_size, bias = True)    # input weight for input gate
        self.Wif_b = nn.Linear(input_size, hidden_size, bias = True)    # input weight for forget gate
        self.Wic_b = nn.Linear(input_size, hidden_size, bias = True)    # input weight for cell gate
        self.Wio_b = nn.Linear(input_size, hidden_size, bias = True)    # input weight for output gate

        self.Whi_b = nn.Linear(hidden_size, hidden_size, bias = True)   # hidden weight for input gate
        self.Whf_b = nn.Linear(hidden_size, hidden_size, bias = True)   # hidden weight for forget gate
        self.Whc_b = nn.Linear(hidden_size, hidden_size, bias = True)   # hidden weight for cell gate
        self.Who_b = nn.Linear(hidden_size, hidden_size, bias = True)   # hidden weight for output gate

    def forward(self, input, hidden):

        # recurrence helper, function for each time step
        def recurrence(input, hidden):
            # previous states
            hx, cx = hidden # hx, cx: (num_layers, batch_size, hidden_size)

            pdb.set_trace()
            # calculate four gates
            input_gate = self.Wii_b(input) + self.Whi_b(hx)
            forget_gate = self.Wif_b(input) + self.Whf_b(hx)
            cell_gate = self.Wic_b(input) + self.Whc_b(hx)
            output_gate = self.Wio_b(input) + self.Who_b(hx)

            # apply non-linearity function
            input_gate = F.sigmoid(input_gate)
            forget_gate = F.sigmoid(forget_gate)
            cell_gate = F.tanh(cell_gate)
            output_gate = F.sigmoid(output_gate)

            # current states
            cy = forget_gate * cx + input_gate * cell_gate
            hy = output_gate * F.tanh(cy)

            # return four gate values for t = seq_len; similar to h_n and c_n
            return (hy, cy), \
                   {'input_gate' : input_gate, 'forget_gate' : forget_gate,
                    'cell_gate' : cell_gate, 'output_gate' : output_gate}

        if self.batch_first:
            input = input.transpose(0, 1)

        output = [] # a list, call cat() before returned
        gates = {}  # a dictionary to receive gate values
        steps = range(input.size(0))    # seq_len

        for i in steps: # process for each tiem step
            hidden, gates = recurrence(input[i], hidden)
            if isinstance(hidden, tuple):
                output.append(hidden[0])
            else:
                output.append(hidden)

        output = torch.cat(output, 0)

        if self.batch_first:
            output = output.transpose(0, 1)

        return output, hidden, gates    # hidden: (h_n, c_n)