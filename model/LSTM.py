import torch
import torch.nn as nn
import torch.nn.functional as F

import pdb

class LSTM(nn.Module):

    def __init__(self, input_size, hidden_size, num_layers = 1, batch_first = True):
        super(LSTM, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.batch_first = batch_first

        # note that nn.Linear contains bias
        self.Wii_b = nn.ModuleList( # input weight for input gate
            [nn.Linear(input_size, hidden_size, bias = True)] +
            [nn.Linear(hidden_size, hidden_size, bias = True) for _ in range(num_layers - 1)])
        self.Wif_b = nn.ModuleList( # input weight for forget gate
            [nn.Linear(input_size, hidden_size, bias=True)] +
            [nn.Linear(hidden_size, hidden_size, bias = True) for _ in range(num_layers - 1)])
        self.Wic_b = nn.ModuleList( # input weight for cell gate
            [nn.Linear(input_size, hidden_size, bias=True)] +
            [nn.Linear(hidden_size, hidden_size, bias = True) for _ in range(num_layers - 1)])
        self.Wio_b = nn.ModuleList( # input weight for output gate
            [nn.Linear(input_size, hidden_size, bias=True)] +
            [nn.Linear(hidden_size, hidden_size, bias = True) for _ in range(num_layers - 1)])

        self.Whi_b = nn.ModuleList( # hidden weight for input gate
            [nn.Linear(hidden_size, hidden_size, bias = True) for _ in range(num_layers)])
        self.Whf_b = nn.ModuleList( # hidden weight for forget gate
            [nn.Linear(hidden_size, hidden_size, bias = True) for _ in range(num_layers)])
        self.Whc_b = nn.ModuleList( # hidden weight for cell gate
            [nn.Linear(hidden_size, hidden_size, bias = True) for _ in range(num_layers)])
        self.Who_b = nn.ModuleList( # hidden weight for output gate
            [nn.Linear(hidden_size, hidden_size, bias = True) for _ in range(num_layers)])

    def forward(self, input, hidden):

        # recurrence helper, function for each time step
        def recurrence(input, hidden):
            # previous hidden states
            hx, cx = hidden # hx, cx: (num_layers, batch_size, hidden_size)

            # store four gate values in all layers for t = seq_len (last time step)
            all_input_gate, all_forget_gate, all_cell_gate, all_output_gate = [], [], [], []

            # for each layer
            for i in range(self.num_layers):
                input_gate = self.Wii_b[i](input) + self.Whi_b[i](hx[i])
                forget_gate = self.Wif_b[i](input) + self.Whf_b[i](hx[i])
                cell_gate = self.Wic_b[i](input) + self.Whc_b[i](hx[i])
                output_gate = self.Wio_b[i](input) + self.Who_b[i](hx[i])

                # apply non-linearity function
                input_gate = F.sigmoid(input_gate)
                forget_gate = F.sigmoid(forget_gate)
                cell_gate = F.tanh(cell_gate)
                output_gate = F.sigmoid(output_gate)

                # current states
                cy = forget_gate * cx[i] + input_gate * cell_gate
                hy = output_gate * F.tanh(cy)

                if self.num_layers == 1:
                    hx = hy.unsqueeze(0)    # (batch_size, hidden_size) -> (1, batch_size, hidden_size)
                    cx = cy.unsqueeze(0)
                elif self.num_layers > 1:
                    # update hx and cx in current layer; avoid in-place operation
                    if i == 0:
                        hx = torch.cat((hy.unsqueeze(0), hx[(i + 1)::, :, :]), 0)
                        cx = torch.cat((cy.unsqueeze(0), cx[(i + 1)::, :, :]), 0)
                    elif i == self.num_layers - 1:
                        hx = torch.cat((hx[0:i, :, :], hy.unsqueeze(0)), 0)
                        cx = torch.cat((cx[0:i, :, :], cy.unsqueeze(0)), 0)
                    else:
                        hx = torch.cat((hx[0::i, :, :], hy.unsqueeze(0), hx[(i + 1)::, :, :]), 0)
                        cx = torch.cat((cx[0::i, :, :], cy.unsqueeze(0), cx[(i + 1)::, :, :]), 0)
                else:
                    raise Exception('Number of layers should be larger than or equal to 1.')

                # upward to upper layer
                input = hy

                # store four gate values in current layer
                all_input_gate.append(input_gate)
                all_forget_gate.append(forget_gate)
                all_cell_gate.append(cell_gate)
                all_output_gate.append(output_gate)

            return (hx, cx), \
                   {'input_gate' : all_input_gate, 'forget_gate' : all_forget_gate,
                    'cell_gate' : all_cell_gate, 'output_gate' : all_output_gate}

        if self.batch_first:
            input = input.transpose(0, 1)

        gates = {}  # a dictionary to receive gate values
        output = [] # a list; call cat() before returned
        steps = range(input.size(0))    # seq_len

        for i in steps: # process for each time step
            hidden, gates = recurrence(input[i], hidden)
            if isinstance(hidden, tuple):
                output.append(hidden[0][self.num_layers - 1].clone().unsqueeze(0))
            else:
                output.append(hidden[self.num_layers - 1].clone().unsqueeze(0))

        output = torch.cat(output, 0)

        if self.batch_first:
            output = output.transpose(0, 1)

        return output, hidden, gates    # hidden: (h_n, c_n)
