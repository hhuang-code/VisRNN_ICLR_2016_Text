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
    for i in range(batch_size): # for every sample in this batch
        start_idx = random.randint(0, file_length - seq_length) # a random number
        end_idx = start_idx + seq_length + 1
        seq = file[start_idx: end_idx]  # seq_length + 1
        input[i] = text_to_tensor(seq[:-1], vocab)  # exclude the last element
        target[i] = text_to_tensor(seq[1:], vocab)  # exclude the first element

    input = Variable(input)
    target = Variable(target)

    # ship to gpu variable
    if torch.cuda.is_available() and config.cuda:
        input = input.cuda()
        target = target.cuda()

    return input, target

def train(inp, target):
    hidden = decoder.init_hidden(args.batch_size)
    if args.cuda:
        hidden = hidden.cuda()
    decoder.zero_grad()
    loss = 0

    for c in range(args.seq_length):
        output, hidden = decoder(inp[:,c], hidden)
        loss += criterion(output.view(args.batch_size, -1), target[:,c])

    loss.backward()
    decoder_optimizer.step()

    return loss.data[0] / args.seq_length

def save():
    save_filename = os.path.splitext(os.path.basename(args.filename))[0] + '.pt'
    torch.save(decoder, save_filename)
    print('Saved as %s' % save_filename)

if __name__ == '__main__':

    config = get_config()

    file, vocab = read_file(path.join(config.data_dir, 'input.txt'))

    # initialize models and start training
    decoder = CharRNN(len(vocab), config.hidden_size, len(vocab), model = config.model, n_layers = config.n_layers)

    decoder_optimizer = torch.optim.Adam(decoder.parameters(), lr = config.learning_rate)

    criterion = nn.CrossEntropyLoss()

    if torch.cuda.is_available() and config.cuda:
        decoder.cuda()  # use gpu

    start = time.time()
    all_losses = []
    loss_avg = 0

    try:
        print("Training for %d epochs..." % config.n_epochs)
        for epoch in tqdm(range(1, config.n_epochs + 1)):
            loss = train(*get_train_set(file, vocab, config.seq_length, config.batch_size))
            loss_avg += loss

            if epoch % args.print_every == 0:
                print('[%s (%d %d%%) %.4f]' % (time_since(start), epoch, epoch / args.n_epochs * 100, loss))
                print(generate(decoder, 'Wh', 100, cuda=args.cuda), '\n')

        print("Saving...")
        save()

    except KeyboardInterrupt:
        print("Saving before quit...")
        save()

