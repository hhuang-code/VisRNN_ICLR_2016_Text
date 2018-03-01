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