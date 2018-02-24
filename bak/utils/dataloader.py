import torch
import pickle

import gc
import math
import re
import operator
import os
import os.path as path

from collections import OrderedDict

import pdb

class CharSplitLMMinibatchLoader:

    def __init__(self):
        pass

    @staticmethod
    def create(data_dir, batch_size, seq_length, split_fractions):

        self = {}

        input_file = path.join(data_dir, 'input.txt')
        vocab_file = path.join(data_dir, 'vocab.pkl')
        tensor_file = path.join(data_dir, 'data.pkl')

        # fetch file attributes to determine if we need to return preprocessing
        run_prepo = False
        if not (path.exists(vocab_file) and path.exists(tensor_file)):
            # prepro files do not exists, generate them
            print('vocab.pkl and data.pkl do not exists. Running preprocessing...')
            run_prepo = True
        else:
            # check if the input file was modifed since last time we ran the prepo.
            # If so, we have to re-run the preprocessing
            input_attr = os.stat(input_file)
            vocab_attr = os.stat(vocab_file)
            tensor_attr = os.stat(tensor_file)
            if input_attr.st_mtime > vocab_attr.st_mtime or input_attr.st_mtime > tensor_attr.st_mtime:
                print('vocab.pkl or data.pkl detected as stale. Re-running preprocessing...')
                run_prepo = True

        if run_prepo:
            # construct a tensor with all the data, and vocab file
            print('one-time setup: preprocessing input text file ', input_file, '...')
            CharSplitLMMinibatchLoader.text_to_tensor(input_file, vocab_file, tensor_file)

        print('loading data files...')
        with open(tensor_file, 'rb') as f:
            data = pickle.load(f)
        f.close()
        with open(vocab_file, 'rb') as f:
            self['vocab_mapping'] = pickle.load(f)
        f.close()

        # cutoff the end so that it divides evenly
        length = data.shape[0]
        if length % (batch_size * seq_length) != 0:
            print('cutting off end of data so that the batches/sequences divide evenly')
            data = data[0: batch_size * seq_length * math.floor(length / (batch_size * seq_length))]

        # count vocab
        self['vocab_size'] = 0
        for _ in self['vocab_mapping']:
            self['vocab_size'] = self['vocab_size'] + 1

        # self['batches'] is a table of tensors
        print('reshaping tensor...')
        self['batch_size'] = batch_size
        self['seq_length'] = seq_length

        ydata = data.clone()
        ydata[0: -1].copy_(data[1:])
        ydata[-1] = data[0]
        self['x_batches'] = data.view(batch_size, -1).split(seq_length, 1)  # |rows| = |batches|
        self['nbatches'] = len(self['x_batches'])
        self['y_batches'] = ydata.view(batch_size, -1).split(seq_length, 1) # |rows| = |batches|
        assert len(self['x_batches']) == len(self['y_batches'])

        # let's try to be helpful here
        if self['nbatches'] < 50:
            print('WARNING: than 50 batches in the data in total? Looks like very small dataset. '
                  'You probably want to use smaller batch_size and/or seq_length.')

        # perform safety checks on split_fractions
        assert split_fractions['train_frac'] >= 0 and split_fractions['train_frac'] <= 1, \
            'bad split fraction ' + str(split_fractions['train_frac']) + ' for train, not between 0 and 1'
        assert split_fractions['val_frac'] >= 0 and split_fractions['val_frac'] <= 1, \
            'bad split fraction ' + str(split_fractions['val_frac']) + ' for val, not between 0 and 1'
        assert split_fractions['test_frac'] >= 0 and split_fractions['test_frac'] <= 1, \
            'bad split fraction ' + str(split_fractions['test_frac']) + ' for test, not between 0 and 1'

        if split_fractions['test_frac'] == 0:
            # catch a common special case where the user might not want a test set
            self['ntrain'] = math.floor(self['nbatches'] * split_fractions['train_frac'])
            self['nval'] = self['nbatches'] - self['ntrain']
            self['ntest'] = 0
        else:
            # divide data to train/val and allocate the rest to test
            self['ntrain'] = math.floor(self['nbatches'] * split_fractions['train_frac'])
            self['nval'] = math.floor(self['nbatches'] * split_fractions['val_frac'])
            self['ntest'] = self['nbatches'] - self['nval'] - self['ntrain']    # the rest goes to test

        self['split_sizes'] = {'ntrain': self['ntrain'], 'nval': self['nval'], 'ntest': self['ntest']}
        self['batch_ix'] = [0, 0, 0]

        print('Data load done! Number of data batches in train: %d, val: %d, test: %d' %
              (self['ntrain'], self['nval'], self['ntest']))

        gc.collect()

        return self

    @staticmethod
    def text_to_tensor(in_textfile, out_vocabfile, out_tensorfile):
        print('loading text file...')

        cache_len = 3
        tot_len = 0
        with open(in_textfile, 'r') as f:
            # create vocabulary if it doesn't exist yet
            print('creating vocabulary mapping...')
            # record all characters to a set
            unordered = {}
            while True:
                rawdata = f.read(cache_len)
                if not rawdata:
                    break   # end of file
                for char in re.compile(r'[\s\S]').findall(rawdata):
                    if char not in unordered:
                        unordered[char] = True
                tot_len = tot_len + len(rawdata)
        f.close()

        # sort into a table
        ordered = OrderedDict()
        for char in unordered.keys():
            ordered[len(ordered) + 1] = char
        ordered = OrderedDict(sorted(ordered.items(), key = operator.itemgetter(1))) # sort by value

        # invert 'ordered' to create the char->int mapping
        vocab_mapping = OrderedDict()
        for i, char in ordered.items():
            vocab_mapping[char] = i

        # construct a tensor with all data
        print('putting data into tensor...')
        data = torch.ByteTensor(tot_len)    # store in into 1D first, then rearrange
        with open(in_textfile, 'r') as f:
            currlen = 0
            while True:
                rawdata = f.read(cache_len)
                if not rawdata:
                    break   # end of file
                for i in range(len(rawdata)):
                    data[currlen + i] = vocab_mapping[rawdata[i]]
                currlen += len(rawdata)
        f.close()

        # save output preprocessed files
        print('saving ', out_vocabfile)
        f = open(out_vocabfile, 'wb')
        pickle.dump(vocab_mapping, f)
        f.close()
        print('saving ', out_tensorfile)
        f = open(out_tensorfile, 'wb')
        pickle.dump(data, f)
        f.close()