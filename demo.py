from utils import *
from Config import *

from collections import OrderedDict

import pdb

if __name__ == '__main__':

    config = get_config()

    test_frac = max(0, 1 - (config.train_frac + config.val_frac))
    split_sizes = OrderedDict()
    split_sizes['train_frac'] = config.train_frac
    split_sizes['val_frac'] = config.val_frac
    split_sizes['test_frac'] = test_frac

    # create the data loader
    loader = CharSplitLMMinibatchLoader.create(config.data_dir, config.batch_size, config.seq_length, split_sizes)
