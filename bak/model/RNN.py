from utils import *

class RNN:
    def __init__(self):
        pass

    @staticmethod
    def rnn(input_size, rnn_size, n, dropout):
        # there are n + 1 inputs (hiddens on each layer and x)
        inputs = {}
        inputs['x'] = nn.Identity()

        outputs = {}
        for L in range(n):
            prev_h = inputs[L + 1]
            if L == 0:
                x = OneHot(input_size)(inputs[0])
                input_size_L = input_size
            else:
                x = outputs[(L - 1)]
                if dropout > 0:
                    x = nn.Dropout(dropout)(x)  # apply dropout, if any
                input_size_L = rnn_size

            # rnn tick
            i2h = nn.Linear(input_size_L, rnn_size)(x)
            h2h = nn.Linear(rnn_size, rnn_size)(prev_h)
            next_h = nn.Tanh()(i2h + h2h)

            outputs['next_h'] = next_h

            # setup the decoder
            top_h = outputs[len(outputs)]
            if dropout > 0:
                top_h = nn.Dropout(dropout)(top_h)
            proj = nn.Linear(rnn_size, input_size)(top_h)
            logsoft = nn.LogSoftmax()(proj)
