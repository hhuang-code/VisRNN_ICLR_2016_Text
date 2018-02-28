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
    parser.add_argument('--data_dir', type = str, default = '/localdisk/hh/dataset/text/shakespeare',
                        help = 'data directory. Should contain the file input.txt with input data')
    parser.add_argument('--input_file', type = str, default = 'text', help = 'input text. Should be in ASCII format.')

    # model params
    parser.add_argument('--model', type = str, default = "lstm", help = 'lstm or gru')
    parser.add_argument('--hidden_size', type = int, default = 128, help = 'size of RNN internal state')
    parser.add_argument('--n_layers', type = int, default = 1, help = 'number of layers in the RNN')

    # optimiaztion
    parser.add_argument('--learning_rate', type = float, default = 0.002, help = 'learning rate')
    parser.add_argument('--seq_length', type = int, default = 70, help = 'number of timesteps to unroll for')
    parser.add_argument('--batch_size', type = int, default = 100, help = 'number of sequences to train on in parallel')
    parser.add_argument('--max_epochs', type = int, default = 50, help = 'number of full passes through the training data')
    parser.add_argument('--train_frac', type = float, default = 0.8, help = 'fraction of data that goes into train set')
    parser.add_argument('--val_frac', type = float, default = 0.1, help = 'fraction of data that goes into validation set')

    # bookkeeping
    parser.add_argument('--print_interval', type = int, default = 10, help = 'how many steps/minibatches between printing out the loss')
    parser.add_argument('--model_dir', type = str, default = '/localdisk/hh/models')

    # gpu
    parser.add_argument('--cuda', action = 'store_true', default = True, help = 'whether to use gpu acceleration')

    args = parser.parse_args()

    # namespace -> dictionary
    args = vars(args)
    args.update(kwargs)

    return Config(**args)