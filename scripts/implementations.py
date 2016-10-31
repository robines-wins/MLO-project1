# -*- coding: utf-8 -*-

"""Implementation of the methods we have seen in the class and the labs"""

from functools import partial

from helpers import *


def compute_cost(y, tx, w):
    """Compute the costs, can use MSE or MAE"""
    return compute_cost_MSE(y, tx, w)


def compute_cost_MSE(y, tx, w):
    """Compute the costs using MSE"""
    e = y - (tx @ w)
    return (1 / (2 * y.shape[0])) * (e.T @ e)


def compute_cost_MAE(y, tx, w):
    """Compute the costs using MAE"""
    e = y - (tx @ w)
    return (1 / y.shape[0]) * np.absolute(e).sum()


def compute_gradient_MSE(y, tx, w):
    """Compute the gradient using MSE"""
    e = y - (tx @ w)
    return -1 / y.shape[0] * (tx.T @ e)


def sigmoid(t):
    """apply sigmoid function on t."""
    z = np.exp(t)
    return z / (1 + z)


def compute_loss_logistic(y, tx, w, lambda_=0):
    """compute the cost by negative log likelihood."""
    clip = np.clip(tx @ w, -700, 700)
    if lambda_ == 0:
        return (1 / w.shape[0]) * np.sum(np.log(1 + np.exp(clip)) - y * (tx @ w))
    else:
        return (1 / w.shape[0]) * np.sum(np.log(1 + np.exp(clip)) - y * (tx @ w)) + lambda_ * np.sum(w * w)
        # or + lambda_*w.T*w
        # return -np.sum(np.log(1+np.exp(tx @ w))-y*(tx @ w))


def compute_gradient_sigmoid(y, tx, w, lambda_=0):
    """compute the gradient of loss."""
    clip = np.clip(tx @ w, -700, 700)
    if lambda_ == 0:
        return (1 / y.shape[0]) * tx.T.dot(sigmoid(clip) - y)
    else:
        return (1 / y.shape[0]) * tx.T.dot(sigmoid(clip) - y) + lambda_ * 2 * w


def general_gradient_descent(y, tx, initial_w, max_iters, gamma, grad_function, cost_function):
    """Gradient descent algorithm who work with arbitrary gradient and cost function
    grad and cost function should take y,tw and w as parameter and return respectively the gradient vector and the
    scalar error"""

    # Define parameters to store w and loss
    ws = [initial_w.ravel()]
    losses = []
    w = initial_w.ravel()

    for n_iter in range(max_iters):
        # compute gradient and loss
        gradient = grad_function(y, tx, w)
        loss = cost_function(y, tx, w)

        # update w by gradient
        w = w - gamma * gradient

        # store w and loss
        ws.append(np.copy(w))
        losses.append(loss)

    return losses[-1], ws[-1]


def general_stochastic_gradient_descent(y, tx, initial_w, batch_size, max_iters, gamma, stock_grad_function,
                                        cost_function):
    """Gradient descent algorithm who work with arbitrary gradient and cost function
    grad and cost function should take y,tw and w as parameter and return respectively the gradient vector and the
    scalar error"""

    # implement stochastic gradient descent.
    ws = [initial_w.ravel()]
    losses = []
    w = initial_w.ravel()

    minibatchs = batch_iter(y, tx, batch_size, num_batches=np.math.floor(y.shape[0] / batch_size))
    for n_iter in range(0, max_iters):
        try:
            minibatch = minibatchs.__next__()

            # compute gradient and loss
            gradient = stock_grad_function(minibatch[0], minibatch[1], w)
            loss = cost_function(y, tx, w)

            # update w by gradient
            w = w - gamma * gradient

            # store w and loss
            ws.append(np.copy(w))
            losses.append(loss)
        except StopIteration:
            return losses[-1], ws[-1]

    return losses[-1], ws[-1]


def least_squares_GD(y, tx, initial_w, max_iters, gamma):
    """Compute the least square using Gradient Descent algorithm"""
    return general_gradient_descent(y, tx, initial_w, max_iters, gamma, compute_gradient_MSE,
                                    compute_cost_MSE)


def least_squares_SGD(y, tx, initial_w, max_iters, gamma):
    """Compute the least square using Stochastic Gradient Descent algorithm"""
    batch_size = y.shape[0] // 2
    return general_stochastic_gradient_descent(y, tx, initial_w, batch_size, max_iters, gamma,
                                               compute_gradient_MSE, compute_cost_MSE)


def least_squares(y, tx):
    """Compute the least square"""
    w = np.linalg.solve(tx.T.dot(tx), tx.T.dot(y))  # return best weight
    return compute_cost_MSE(y, tx, w), w


def ridge_regression(y, tx, lambda_):
    """implement ridge regression."""
    lambp = lambda_ * (2 * tx.shape[0])
    w = np.linalg.solve(tx.T.dot(tx) + lambp * np.eye(tx.shape[1]), tx.T.dot(y))
    return compute_cost_MSE(y, tx, w), w


def logistic_regression(y, tx, initial_w, max_iters, gamma):
    """Implement logistic regression using the gradient descent algorithm"""
    ok = general_gradient_descent(y, tx, initial_w, max_iters, gamma, compute_gradient_sigmoid, compute_loss_logistic)
    return ok


def reg_logistic_regression(y, tx, lambda_, initial_w, max_iters, gamma):
    """Implement regular logistic regression using the gradient descent algorithm"""
    gradf = partial(compute_gradient_sigmoid, lambda_=lambda_)
    costf = partial(compute_loss_logistic, lambda_=lambda_)
    return general_gradient_descent(y, tx, initial_w, max_iters, gamma, gradf, costf)
