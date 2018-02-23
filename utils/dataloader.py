import torch
import pickle

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

        input_file = path.join(data_dir, 'warpeace_input.txt')
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
        len = data.shape[0]
        if len % (batch_size * seq_length) != 0:
            print('cutting off end of data so that the batches/sequences divide evenly')
            data = data[0: batch_size * seq_length * math.floor(len / (batch_size * seq_length))]

        # count vocab
        self['vocab_size'] = 0
        for _ in self['vocab_mapping']:
            self['vocab_size'] = self['vocab_size'] + 1

        # self['batches'] is a table of tensors
        print('reshaping tensor...')
        self['batch_size'] = batch_size
        self['seq_length'] = seq_length


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