import torch

import numpy as np

import pdb

# remove outliers from a 1D array; alpha set to 1.5 by default
# return the index of outliers
def remove_outliers(arr, alpha = 1.5):
    q1 = np.percentile(arr, 25)
    q3 = np.percentile(arr, 75)
    iqr = q3 - q1
    valid_range = (q1 - alpha * iqr, q3 + alpha * iqr)
    outliers_idx = []
    for i in range(len(arr)):
        if arr[i] > valid_range[1] or arr[i] < valid_range[0]:
            outliers_idx.append(i)

    return outliers_idx


# convert each character from int to one-hot vector
'''
input: a tensor, shape (batch_size, seq_len)
return: a tensor, shape (batch_size, seq_len, input_size)
'''
def convert_to_onehot(input_tensor, input_size):
    input_shape = input_tensor.shape
    input_tensor = torch.unsqueeze(input_tensor, 2)   # (batch_size, seq_len) -> (batch_size, seq_len, 1)
    onehot_tensor = torch.zeros(input_shape[0], input_shape[1], input_size)
    onehot_tensor.scatter_(2, input_tensor, 1)  # (batch_size, seq_len, input_size)

    return onehot_tensor