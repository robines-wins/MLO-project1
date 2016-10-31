# -*- coding: utf-8 -*-

"""functions used to crossvalidate part of our training set with the other part
   and search for optimal parameters or function to use"""
from functools import partial

import numpy as np


def build_k_indices(y, k_fold, seed):
    """build k indices for k-fold."""
    num_row = y.shape[0]
    interval = int(num_row / k_fold)
    np.random.seed(seed)
    indices = np.random.permutation(num_row)
    k_indices = [indices[k * interval: (k + 1) * interval] for k in range(k_fold)]
    return np.array(k_indices)


def cross_validation(y, tx, gradient_descent, initial_w, max_iters, gamma, grad_function, cost_function, k_indices, k,
                     lambda_):
    """return the loss of train values, loss of test values and weights"""
    # get k'th subgroup in test, others in train:
    train_indices = k_indices[[i for i in range(len(k_indices)) if i != k]]
    train_tx, train_y = x_poly[np.ravel(train_indices)], y[np.ravel(train_indices)]
    test_tx, test_y = x_poly[k_indices[k]], y[k_indices[k]]

    gradf = partial(grad_function, lambda_=lambda_)
    costf = partial(cost_function, lambda_=lambda_)

    loss_tr, weight = gradient_descent(y, tx, initial_w, max_iters, gamma, grad_function, cost_function)

    loss_te = compute_loss(test_y, test_tx, weight)
    return loss_tr, loss_te, weight
