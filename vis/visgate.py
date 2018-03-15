import os.path as path
import numpy as np

from model import *
from utils import *

import pdb

def vis_gate(test_set, vocab_size, config):
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
    test_batch_idx = np.random.choice(test_input_set.shape[0])  # random batch index
    test_seq_idx = np.random.choice(config.batch_size)  # random sequence index
    warmup_seq = test_input_set[test_batch_idx][test_seq_idx].unsqueeze(0)  # random sequence

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

    input_gate = [] # store values of input gate for each time step; row per time step, and col per entry in vector
    forget_gate = []    # store values of forget gate for each time step
    output_gate = []    # store values of output gate for each time step
    stop_flag = False  # flag to stop
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

            # (seq_len) -> (1, seq_len)
            test_seq = test_seq.view(1, -1)

            # ship to gpu if possible
            if torch.cuda.is_available() and config.cuda:
                test_seq = test_seq.cuda()

            # view one sequence as a batch
            for i in range(config.seq_length):  # for every time step in this batch
                # forward pass, we do not care about output
                _, hidden, gates = char_rnn(Variable(test_seq[:, i]), hidden)
                # store gate value
                input_gate.append(gates['input_gate_n'].data.cpu().view(1, -1))
                forget_gate.append(gates['forget_gate_n'].data.cpu().view(1, -1))
                output_gate.append(gates['output_gate_n'].data.cpu().view(1, -1))

            # print progress information
            print('Processing [batch: %d, sequence: %3d]...' % (test_batch_idx, test_seq_idx))

    # visualize gate value
    left_thresh = 0.1  # left saturated threshold
    right_thresh = 0.9  # right saturated threshold

    def get_saturated(gate, left_thresh, right_thresh):
        total_seq_length = len(gate)  # total_seq_length = total character number
        gate = torch.cat(gate, 0).numpy()  # (total_seq_length, hidden_size)
        left_s = gate < left_thresh
        right_s = gate > right_thresh
        left_s = np.sum(left_s, 0) / total_seq_length  # boradcasting
        right_s = np.sum(right_s, 0) / total_seq_length

        return left_s, right_s  # left_s/right_s: (hidden_size)

    input_left_s, input_right_s = get_saturated(input_gate, left_thresh, right_thresh)
    forget_left_s, forget_right_s = get_saturated(forget_gate, left_thresh, right_thresh)
    output_left_s, output_right_s = get_saturated(output_gate, left_thresh, right_thresh)

    # saturation plot
    plot_gate((input_left_s, input_right_s), (forget_left_s, forget_right_s), (output_left_s, output_right_s))





