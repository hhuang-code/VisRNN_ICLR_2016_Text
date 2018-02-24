#!/usr/bin/env python
# https://github.com/spro/char-rnn.pytorch

import torch
import torch.nn as nn
from torch.autograd import Variable
import argparse
import os.path as path

from tqdm import tqdm

import pdb

from helpers import *
from model import *
from generate import *

from config import *

# generate training set randomly
def get_train_set(file, vocab, seq_length, batch_size):
    # two tensors: (batch_size * seq_length)
    input = torch.LongTensor(batch_size, seq_length)
    target = torch.LongTensor(batch_size, seq_length)

    file_length = len(file)
    for i in range(batch_size): # for every sample (row) in this batch
        start_idx = random.randint(0, file_length - seq_length) # a random number
        end_idx = start_idx + seq_length
        seq = file[start_idx: end_idx]  # len: seq_length
        # input: (a1, a2, ..., an-1, an) -> target: (a2, ..., an-1, an, a1)
        input[i] = text_to_tensor(seq, vocab)
        target[i][0: -1] = input[i][1:].clone()
        target[i][-1] = input[i][0]

    input = Variable(input)
    target = Variable(target)

    # ship to gpu variable
    if torch.cuda.is_available() and config.cuda:
        input = input.cuda()
        target = target.cuda()

    return input, target    # both of them: (batch_size, seq_length)

def train(input, target):
    init_hidden = char_rnn.init_hidden(config.batch_size) # (n_layers * n_directions, batch_size, hidden_size)

    if torch.cuda.is_available() and config.cuda:
        init_hidden = init_hidden.cuda()

    char_rnn.zero_grad()

    # for i in range(config.seq_length):
    #     output, (h_n, c_n) = decoder(input[:, i], init_hidden)
    #     loss += criterion(output.view(config.batch_size, -1), target[:,i])

    output, _ = char_rnn(input, init_hidden)

    loss = 0
    for i in range(config.batch_size):  # for every sample in the batch
        loss += criterion(output[i], target[i])

    loss.backward()
    decoder_optimizer.step()

    return loss

def save():
    save_filename = os.path.splitext(os.path.basename(config.filename))[0] + '.pt'
    torch.save(decoder, save_filename)
    print('Saved as %s' % save_filename)

if __name__ == '__main__':

    config = get_config()

    file, vocab = read_file(path.join(config.data_dir, 'input.txt'))

    # initialize models and start training
    char_rnn = CharRNN(len(vocab), config.hidden_size, len(vocab), model = config.model, n_layers = config.n_layers)

    decoder_optimizer = torch.optim.Adam(char_rnn.parameters(), lr = config.learning_rate)

    criterion = nn.CrossEntropyLoss()

    if torch.cuda.is_available() and config.cuda:
        decoder.cuda()  # use gpu

    start_time = time.time()

    try:
        print("Training for %d epochs..." % config.n_epochs)
        for epoch in tqdm(range(1, config.n_epochs + 1)):
            loss = train(*get_train_set(file, vocab, config.seq_length, config.batch_size))

            if epoch % config.print_every == 0:
                # print('[%s (%d %d%%) %.4f]' % (elapsed_time(start_time), epoch, epoch / config.n_epochs * 100, loss))
                # print(generate(decoder, 'Wh', 100, cuda=args.cuda), '\n')
                print(loss)

        print("Saving...")
        save()

    except KeyboardInterrupt:
        print("Saving before quit...")
        save()

