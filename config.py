import argparse

import pdb

"""
set configuration arguments as class attributes
"""
class Config(object):

    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)


"""
get configuration arguments
"""
def get_config(**kwargs):

    parser = argparse.ArgumentParser()

    # Parse command line arguments
    parser = argparse.ArgumentParser()
    # data
    parser.add_argument('--data_dir', type = str, default = '/localdisk/dataset/text/shakespeare',
                        help = 'data directory. Should contain the file input.txt with input data')
    parser.add_argument('--input_file', type = str, default = 'text', help = 'input text. Should be in ASCII format.')

    # model params
    parser.add_argument('--model', type = str, default = "lstm", help = 'lstm or gru')
    parser.add_argument('--hidden_size', type = int, default = 128, help = 'size of RNN internal state')
    parser.add_argument('--n_layers', type = int, default = 2, help = 'number of layers in the RNN')

    # optimiaztion
    parser.add_argument('--learning_rate', type = float, default = 0.002, help = 'learning rate')
    parser.add_argument('--seq_length', type = int, default = 200, help = 'number of timesteps to unroll for')
    parser.add_argument('--batch_size', type = int, default = 50, help = 'number of sequences to train on in parallel')
    parser.add_argument('--n_epochs', type = int, default = 50, help = 'number of full passes through the training data')
    parser.add_argument('--shuffle', action = 'store_true')

    # bookkeeping
    parser.add_argument('--print_every', type = int, default = 100, help = 'how many steps/minibatches between printing out the loss')

    # gpu
    parser.add_argument('--cuda', action = 'store_true')

    args = parser.parse_args()

    # namespace -> dictionary
    args = vars(args)
    args.update(kwargs)

    return Config(**args)