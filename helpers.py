# https://github.com/spro/char-rnn.pytorch
import torch

import sys
import unidecode
import string
import random
import time
import math

import pdb

# Reading and un-unicode-encoding data

all_characters = string.printable
n_characters = len(all_characters)

# read in file and generate a vocabulary
def read_file(filename):
    try:
        file = unidecode.unidecode(open(filename).read())
    except IOError:
        print("Could not read file: " + filename)
        sys.exit()

    vocab = ''
    all_chars = string.printable    # all printable characters in python
    for ch in all_chars:
        if ch in file:
            vocab += ch # select characters that appear in the file

    return file, vocab

# Turning a string into a tensor
def text_to_tensor(text, vocab):
    text_length = len(text)
    tensor = torch.zeros(text_length).long()
    for i in range(text_length):    # for every character
        try:
            tensor[i] = vocab.index(text[i])
        except:
            continue

    pdb.set_trace()

    return tensor

# Readable time elapsed

def time_since(since):
    s = time.time() - since
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)

