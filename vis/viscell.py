import numpy as np
import matplotlib.pyplot as plt

import string
from functools import partial

from model import *
from util import *

import pdb

# display vis_seq (unchanged mode) to a html file
def display_unchanged(vis_file, vis_seq, i2c):
    content = []
    for idx, seq in vis_seq.items():
        content.append('<h><strong>Test sequence ' + str(idx + 1) + '</strong></h><br>')
        if seq.is_cuda:
            seq = seq.cpu()
        seq = seq.data.numpy()
        seq = [i2c[x] for x in seq]
        for ch in seq:
            content.append(add_style(ch, 'p'))
        content.append('<br><br>')

    # write to file
    write_html(vis_file, ''.join(content))

# display vis_seq (tosame mode) to a html file
def display_tosame(vis_file, vis_seq, i2c):
    content = []
    for idx, seq in vis_seq.items():
        content.append('<h>Test sequence ' + str(idx + 1) + '</h><br>')
        if seq.is_cuda:
            seq = seq.cpu()
        seq = seq.data.numpy()
        seq = [i2c[x] for x in seq]
        for ch in seq:
            content.append(add_style(ch, 'b'))
        content.append('<br><br>')

    # write to file
    write_html(vis_file, ''.join(content))

# display vis_seq (keepone mode) to a html file
def display_keepone(vis_file, vis_seq, i2c, keeped_char):
    content = []
    for idx, seq in vis_seq.items():
        content.append('<h>Test sequence ' + str(idx + 1) + '</h><br>')
        if seq.is_cuda:
            seq = seq.cpu()
        seq = seq.data.numpy()
        seq = [i2c[x] for x in seq]
        for ch in seq:
            if ch == keeped_char:
                content.append(add_style(ch, 'r'))
            else:
                content.append(add_style(ch, 'b'))
        content.append('<br><br>')

    # write to file
    write_html(vis_file, ''.join(content))

# display vis_seq (dichotom mode) to a html file
def display_dichotom(vis_file, vis_seq, i2c, left_char, right_char):
    content = []
    for idx, seq in vis_seq.items():
        content.append('<h>Test sequence ' + str(idx + 1) + '</h><br>')
        if seq.is_cuda:
            seq = seq.cpu()
        seq = seq.data.numpy()
        seq = [i2c[x] for x in seq]
        for ch in seq:
            if ch == left_char:
                content.append(add_style(ch, 'r'))
            elif ch == right_char:
                content.append(add_style(ch, 'b'))
        content.append('<br><br>')

    # write to file
    write_html(vis_file, ''.join(content))

# plot value of cell state of seq_num sequences;
# each cell state of a sequence is represented by a 2D array (row -> time step; col -> entries of cell state)
def draw(vis_cells, desp):

    seq_num = len(vis_cells)    # number of selected sequences

    fig = plt.figure()

    # super title
    fig.suptitle('Cell Values', fontsize = 16)

    # center text
    fig.text(.5, .05, desp, ha = 'center', fontsize = 12)

    i = 0
    for idx, seq_cell in vis_cells.items():
        ax = fig.add_subplot(2, seq_num / 2, i + 1)
        i += 1
        for j in range(seq_cell.shape[0]):    # for each entry in cell state
            time_step = np.linspace(0, len(seq_cell[j]) - 1, num = len(seq_cell[j]))
            value = seq_cell[j]
            ax.plot(time_step, value)
            ax.set_title('Test sequence ' + str((idx + 1)))
            ax.set_xlabel('time step')
            ax.set_ylabel('value')

    plt.subplots_adjust(left = 0.125, bottom = 0.15, right = 0.9, top = 0.9, wspace = 0.2, hspace = 0.3)
    plt.show()

# analyze value of cells
# def analysis(test_cells):
#     # sum along time axis
#     sum_test_cells = [np.sum(x, axis = 1) for x in test_cells]
#
#     hidden_size = test_cells[0].shape[0]
#     index = [int(x) for x in np.linspace(0, hidden_size - 1, hidden_size)]
#
#     # detect outliers
#     test_outliers_idx = []
#     deleted_idx = []
#     for cell in sum_test_cells:
#         outliers_idx = remove_outliers(cell)
#         deleted_idx.append(np.setdiff1d(index, outliers_idx))
#
#     # delete outlier cell
#     for i in range(len(test_cells)):
#         test_cells[i] = np.delete(test_cells[i], deleted_idx[i], axis = 0)    # delete several rows
#
#     return test_cells

# leave test_set unchanged
def leave_unchanged(test_set):
    # unchanged
    return test_set

# change all character in a test_set to the same one, say 'a'
def change_to_same(test_set, c2i):
    # no need to modify test_target_set
    test_input_set, _ = test_set[0], test_set[1]

    for test_batch_idx in range(test_input_set.shape[0]):
        for test_seq_idx in range(test_input_set[test_batch_idx].shape[0]):
            for ch in range(test_input_set[test_batch_idx][test_seq_idx].shape[0]):
                test_input_set[test_batch_idx][test_seq_idx][ch] = c2i['a']

    return test_input_set, test_set[1]

# change all character in a test_set to the same one, say 'a', except for the 'keeped' character
def keep_one(test_set, c2i, keeped):
    # no need to modify test_target_set
    test_input_set, _ = test_set[0], test_set[1]

    for test_batch_idx in range(test_input_set.shape[0]):
        for test_seq_idx in range(test_input_set[test_batch_idx].shape[0]):
            for ch in range(test_input_set[test_batch_idx][test_seq_idx].shape[0]):
                if test_input_set[test_batch_idx][test_seq_idx][ch] != c2i[keeped]:   # only keep 'keeped' character
                    test_input_set[test_batch_idx][test_seq_idx][ch] = c2i['a']

    return test_input_set, test_set[1]

# randomly split a sequence into two halves, the first part set to 'left' character,
# and the second part set to 'right' character
def dichotom(test_set, c2i, left, right):
    # no need to modify test_target_set
    test_input_set, _ = test_set[0], test_set[1]

    for test_batch_idx in range(test_input_set.shape[0]):
        for test_seq_idx in range(test_input_set[test_batch_idx].shape[0]):
            split = np.random.choice(test_input_set[test_batch_idx][test_seq_idx].shape[0], 1)
            for ch in range(test_input_set[test_batch_idx][test_seq_idx].shape[0]):
                if ch < split[0]:
                    test_input_set[test_batch_idx][test_seq_idx][ch] = c2i[left]
                else:
                    test_input_set[test_batch_idx][test_seq_idx][ch] = c2i[right]

    return test_input_set, test_set[1]

# visualize cell state values for variable test sequence
'''
mode: 
    unchanged - keep test_set unchanged 
    tosame - change all character in test_set to the same character, say 'a'
    keepone - change all character in test_set to the same character, say 'a', except for the 'keeped' character
    dichotom - split a sequence into two halves, the first part set to 'left' character, 
                and the second part set to 'right' character
'''
def vis_cell(test_set, char_to_int, int_to_char, config, mode = 'unchanged'):
    # not existing trained model, train a new one
    if not path.exists(path.join(config.model_dir, config.model + '.pth')):
        raise Exception('No such trained model! Please train a model first!')
        sys.exit(-1)

    # load a trained model
    char_rnn = CharRNN(len(char_to_int), config.hidden_size, len(char_to_int),
                       model = config.model, n_layers = config.n_layers)
    char_rnn.load_state_dict(torch.load(path.join(config.model_dir, config.model + '.pth')))
    char_rnn.eval()

    # initialize hidden state
    init_hidden = char_rnn.init_hidden(1)  # batch_size = 1

    # randomly selected a character to be keeped
    keeped_char =  int_to_char[np.random.choice(len(int_to_char), 1)[0]]
    keeped_char = 'y'
    print('The keeped character: %s' % keeped_char)

    # randomly selected 'left' and 'right' character
    lr_idx = np.random.choice(len(string.ascii_letters), 2)
    left_char, right_char = string.ascii_letters[lr_idx[0]], string.ascii_letters[lr_idx[1]]
    print('The left character: %s; the right character: %s' % (left_char, right_char))

    # switch simulation
    switch = {
        'unchanged': leave_unchanged,
        'tosame': partial(change_to_same, c2i = char_to_int),
        'keepone': partial(keep_one, c2i = char_to_int, keeped = keeped_char),
        'dichotom': partial(dichotom, c2i = char_to_int, left = left_char, right = right_char),
    }

    # modify test_set according to mode
    test_set = switch[mode](test_set)

    # convert test_set from tensor to Variable, and there's no need to convert test_target_set into Variable
    test_input_set, _ = Variable(test_set[0]), test_set[1]

    # ship to gpu if possible
    if torch.cuda.is_available() and config.cuda:
        char_rnn.cuda()
        test_input_set = test_input_set.cuda()  # no need to ship test_target_set to gpu Variable
        init_hidden = tuple([x.cuda() for x in init_hidden])

    # content = []    # an empty string list
    # randomly select seq_num of sequences
    seq_num = 6
    samples = np.random.choice(test_input_set.shape[0] * test_input_set.shape[1], seq_num)
    vis_seqs = dict()  # sequences to be displayed
    vis_cells = dict()  # cell state of sequences to be visualized
    cnt = 0 # a counter
    for test_batch_idx in range(test_input_set.shape[0]):
        # for every batch
        test_batch = test_input_set[test_batch_idx]
        # for every sequence in this batch
        for test_seq_idx in range(test_batch.shape[0]):
            seq_cell = np.array([]) # store cell entry values for current sequence
            test_seq = test_batch[test_seq_idx]
            if cnt in samples:
                vis_seqs[cnt] = test_seq
                # for every character in this sequence
                for ch in test_seq:
                    _, init_hidden = char_rnn(ch.view(1, -1), init_hidden)
                    (h_n, c_n) = init_hidden[0], init_hidden[1]
                    if c_n.is_cuda:
                        tmp_c_n = c_n.cpu()
                    if len(seq_cell) == 0:
                        seq_cell = tmp_c_n.data.squeeze().numpy()
                    else:
                        seq_cell = np.vstack((seq_cell, tmp_c_n.data.squeeze().numpy()))
                # transpose: each row represents an entry of the cell; size: (hidden_size, seq_len)
                vis_cells[cnt] = np.transpose(seq_cell)
            # add counter
            cnt += 1

    # analyze cells values
    # test_cells = analysis(test_cells)

    # sequence visualization
    vis_file = path.join(config.vis_dir, 'vis_seq.html')
    # switch simulation
    switch = {
        'unchanged': partial(display_unchanged, i2c = int_to_char),
        'tosame': partial(display_tosame, i2c = int_to_char),
        'keepone': partial(display_keepone, i2c = int_to_char, keeped_char = keeped_char),
        'dichotom': partial(display_dichotom, i2c = int_to_char, left_char = left_char, right_char = right_char),
    }
    switch[mode](vis_file, vis_seqs)

    # plot raw values of cell state of selected sequences
    draw(vis_cells, 'Raw value of cell state, randomly selected 6 sequences')

    # plot tanh cell values from 6 randomly selected test sequences
    # draw(np.tanh(test_cells), 'Tanh value of cell state, randomly selected 6 sequences', seq_num = 6)

    # plot average cell values from 6 randomly selected test sequences
    draw([np.expand_dims(np.sum(x, axis = 0), axis = 0) for x in test_cells],
         'Average value of cell state, randomly selected 6 sequences', seq_num = 6)
