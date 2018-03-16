import torch
from torch.autograd import Variable

from model import *

import pdb

pdb.set_trace()
rnn = LSTM(10, 20, 2, batch_first = True).cuda()    # (input_size, hidden_size, num_layers)
inputs = Variable(torch.randn(3, 5, 10)).cuda()    # (batch_size, seq_len, input_size)
h0 = Variable(torch.randn(2, 3, 20)).cuda()    # (num_layers, batch_size, hidden_size)
c0 = Variable(torch.rand(2, 3, 20)).cuda()
output, hidden = rnn(inputs, (h0, c0))

# rnn = RNN(10, 20, 1, batch_first = True).cuda()
# inputs = Variable(torch.randn(3, 5, 10)).cuda()
# h0 = Variable(torch.randn(1, 3, 20)).cuda()
# output, hn = rnn(inputs, h0)

pdb.set_trace()
