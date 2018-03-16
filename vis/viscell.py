import json

import os.path as path
import numpy as np

from model import *

import pdb

def vis_cell(test_set, int_to_char, vocab_size, config):
    # no trained model, train a new one
    if not path.exists(path.join(config.model_dir, config.model + '.pth')):
        raise Exception('No such a trained model! Please train a new model first!')

    # load a trained model
    char_rnn = CharRNN(vocab_size, config.hidden_size, vocab_size, model = config.model, n_layers = config.n_layers)
    char_rnn.load_state_dict(torch.load(path.join(config.model_dir, config.model + '.pth')))
    char_rnn.eval()

    # ship to gpu if possible
    if torch.cuda.is_available() and config.cuda:
        char_rnn.cuda()

    # prepare test data
    test_input_set, _ = test_set[0], test_set[1]  # test_input_set: (test_batches, batch_size, seq_length)

    # randomly choose a sequence in test set to warm up the network
    test_batch_idx = np.random.choice(test_input_set.shape[0])    # random batch index
    test_seq_idx = np.random.choice(config.batch_size) # random sequence index
    warmup_seq = test_input_set[test_batch_idx][test_seq_idx].unsqueeze(0)   # random sequence

    # initialize hidden state
    hidden = char_rnn.init_hidden(1)  # here, batch_size = 1

    # ship to gpu if possible
    if torch.cuda.is_available() and config.cuda:
        warmup_seq = warmup_seq.cuda()
        hidden = tuple([x.cuda() for x in hidden])

    # warmup network
    for i in range(config.seq_length):
        # get final hidden state
        _, hidden, _ = char_rnn(Variable(warmup_seq[:, i]), hidden)

    seq = []    # store all test sequences in character form
    cell = []   # 2d array, store all cell state values; each character corresponds to a row; each row is a c_n
    stop_flag = False   # flag to stop
    for test_batch_idx in range(1, test_input_set.shape[0] + 1):

        # whether to stop
        if stop_flag:
            break

        # for every batch
        test_batch = test_input_set[test_batch_idx - 1]
        # for every sequence in this batch
        for test_seq_idx in range(1, config.batch_size + 1):

            # whether to stop
            if (config.batch_size * (test_batch_idx - 1) + test_seq_idx) * config.seq_length > config.max_vis_char:
                stop_flag = True
                break

            # current sequence
            test_seq = test_batch[test_seq_idx - 1]
            # append to seq
            seq.extend([int_to_char[x] for x in test_seq])  # do not use append() function

            # (seq_len) -> (1, seq_len)
            test_seq = test_seq.view(1, -1)

            # ship to gpu if possible
            if torch.cuda.is_available() and config.cuda:
                test_seq = test_seq.cuda()

            # view one sequence as a batch
            for i in range(config.seq_length):  # for every time step in this batch
                # forward pass, we do not care about output
                _, hidden, _ = char_rnn(Variable(test_seq[:, i]), hidden)
                (_, c_n) = hidden   # c_n: (1, 1, hidden_size); ignore h_n
                cell.append(c_n.data.cpu().squeeze().numpy())   # use append() to form a multi-dimensional array

            # print progress information
            print('Processing [batch: %d, sequence: %3d]...' % (test_batch_idx, test_seq_idx))

    # write seq and cell into a json file for visualization
    char_cell = {}
    char_cell['cell_size'] = config.hidden_size
    char_cell['seq'] = ''.join(seq)

    # allocate space for cell values
    for j in range(config.n_layers):
        char_cell['cell_layer_' + str(j + 1)] = []

    total_char = len(cell)
    for i in range(total_char): # for each character (time step)
        for j in range(config.n_layers):   # for each layer
            char_cell['cell_layer_' + str(j + 1)].append(cell[i][j].tolist())

    with open(path.join(config.vis_dir, 'char_cell.json'), 'w') as json_file:
        json.dump(char_cell, json_file)


