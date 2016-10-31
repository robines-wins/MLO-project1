# -*- coding: utf-8 -*-

"""functions used to crossvalidate part of our training set with the other part
   and search for optimal parameters or function to use"""

import matplotlib.pyplot as plt

from costs import *
from implementations import *


def build_k_indices(y, k_fold, seed):
    """build k indices for k-fold cross validation."""
    num_row = y.shape[0]
    interval = int(num_row / k_fold)
    np.random.seed(seed)
    indices = np.random.permutation(num_row)
    k_indices = [indices[k * interval: (k + 1) * interval] for k in range(k_fold)]
    return np.array(k_indices)


def cross_validation_visualization(graph_name, x_name, x_values, mse_tr, mse_te):
    """visualization the curves of mse_tr and mse_te.
       take the name of the x axis and graph name as argument"""
    plt.semilogx(x_values, mse_tr, marker=".", color='b', label='train error')
    plt.semilogx(x_values, mse_te, marker=".", color='r', label='test error')
    plt.xlabel(x_name)
    plt.ylabel("rmse")
    plt.title(graph_name)
    plt.legend(loc=2)
    plt.grid(True)
    plt.savefig(graph_name)


def cross_validation(y, tx, function_to_test, k_fold, lambda_, seed, cost_function):
    """return the loss of train values, loss of test values and weights
       using k-fold cross validation for a function given as parameter"""
    k_indices = build_k_indices(y, k_fold, seed)
    loss_tr = []
    loss_te = []
    weights = []
    for k in range(k_indices.shape[0]):
        # get k'th subgroup in test, others in train:
        train_indices = k_indices[[i for i in range(len(k_indices)) if i != k]]
        train_tx, train_y = tx[np.ravel(train_indices)], y[np.ravel(train_indices)]
        test_tx, test_y = tx[k_indices[k]], y[k_indices[k]]

        # Call the test function with the correct arguments depending on what we received
        if isinstance(function_to_test, partial):
            if (function_to_test.func == reg_logistic_regression):
                loss_tr_k, weight_k = function_to_test(train_y, train_tx, lambda_)
            else:
                loss_tr_k, weight_k = function_to_test(train_y, train_tx)
        elif (function_to_test == ridge_regression):
            loss_tr_k, weight_k = function_to_test(train_y, train_tx, lambda_)
        else:
            loss_tr_k, weight_k = function_to_test(train_y, train_tx)

        # Save the results for this value
        loss_tr.append(loss_tr_k)
        weights.append(weight_k)
        loss_te.append(cost_function(test_y, test_tx, weight_k))
    return np.mean(loss_tr), np.mean(loss_te), np.mean(weights, axis=0)


def finding_lambda(y, tx, function_to_test, k_fold, seed, lambdas, cost_function):
    """Helper function to iterate over a list of different lambda values to compare
       the effect of the lambda parameter"""

    # define lists to store the loss of training data and test data
    loss_tr = []
    loss_te = []
    for lambda_ in lambdas:
        # execute cross_validation for this lambda value
        if (cost_function == compute_loss_logistic):
            cost_function = partial(compute_loss_logistic, lambda_=lambda_)
        loss_tr_lamb, loss_te_lamb, weight_lamb = cross_validation(y, tx, function_to_test, k_fold, lambda_, seed,
                                                                   cost_function)
        loss_tr.append(loss_tr_lamb)
        loss_te.append(loss_te_lamb)
    return loss_tr, loss_te
