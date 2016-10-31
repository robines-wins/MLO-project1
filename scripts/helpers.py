# -*- coding: utf-8 -*-

"""some helper functions."""
import numpy as np


def standardize(x, mean_x=None, std_x=None):
    """Standardize the original data set."""
    if mean_x is None:
        mean_x = np.mean(x, axis=0)
    x = x - mean_x
    if std_x is None:
        std_x = np.std(x, axis=0)
    x[:, std_x > 0] = x[:, std_x > 0] / std_x[std_x > 0]

    tx = np.hstack((np.ones((x.shape[0], 1)), x))
    return tx, mean_x, std_x


def remove_outliers(tx, mean_x, std_x):
    """"Remove the outliers from the initial dataset by setting the value of each feature to it's mean if it is not
    within 2 standard deviation of the mean. In other words we only keep the values within a 95% confidence interval"""
    n_tx = tx.copy()
    for sample in range(tx.shape[0]):
        for dim in range(tx.shape[1]):
            if (n_tx[sample, dim] > mean_x[dim] + 2 * std_x[dim]):
                n_tx[sample, dim] = mean_x[dim]
            if (n_tx[sample, dim] < mean_x[dim] - 2 * std_x[dim]):
                n_tx[sample, dim] = mean_x[dim]
            if (n_tx[sample, dim] == -999):
                n_tx[sample, dim] = 0
    return n_tx


def rescale(tx):
    """Rescale the input data to values between 0 and 1"""
    mins = np.amin(tx, axis=0)
    maxs = np.amax(tx, axis=0)
    txscale = (tx - mins) / (maxs - mins)
    return txscale


def format_y(y):
    """Reformat the y vector to binary format (0 or 1)"""
    y2 = y.copy()
    y2[np.where(y == -1)] = 0
    return y2


def batch_iter(y, tx, batch_size, num_batches=None, shuffle=True):
    """
    Generate a minibatch iterator for a dataset.
    Takes as input two iterables (here the output desired values 'y' and the input data 'tx')
    Outputs an iterator which gives mini-batches of `batch_size` matching elements from `y` and `tx`.
    Data can be randomly shuffled to avoid ordering in the original data messing with the randomness of the minibatches.
    Example of use :
    for minibatch_y, minibatch_tx in batch_iter(y, tx, 32):
        <DO-SOMETHING>
    """
    data_size = len(y)
    num_batches_max = int(np.ceil(data_size / batch_size))
    if num_batches is None:
        num_batches = num_batches_max
    else:
        num_batches = min(num_batches, num_batches_max)

    if shuffle:
        shuffle_indices = np.random.permutation(np.arange(data_size))
        shuffled_y = y[shuffle_indices]
        shuffled_tx = tx[shuffle_indices]
    else:
        shuffled_y = y
        shuffled_tx = tx
    for batch_num in range(num_batches):
        start_index = batch_num * batch_size
        end_index = min((batch_num + 1) * batch_size, data_size)
        if start_index != end_index:
            yield shuffled_y[start_index:end_index], shuffled_tx[start_index:end_index]
