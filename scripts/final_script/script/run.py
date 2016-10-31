
# coding: utf-8

# In[ ]:

# Useful starting lines
import numpy as np
import matplotlib.pyplot as plt
from functools import partial


# # Load the training data into feature matrix, class labels, and event ids:

# In[ ]:


# # Implement ML functions

# In[ ]:

def compute_cost(y, tx, w):
    return compute_cost_MSE(y,tx,w)
    
def compute_cost_MSE(y,tx,w):
    e=y-(tx @ w)
    return (1/(2*y.shape[0]))*(e.T @ e)
def compute_cost_MAE(y,tx,w):
    e=y-(tx @ w)
    return (1/y.shape[0])*np.absolute(e).sum()


# In[ ]:

def compute_gradient_MSE(y,tx,w):
    """Compute the gradient."""
    e=y-(tx @ w)
    return -1/y.shape[0]*(tx.T @ e)


# In[ ]:

def general_gradient_descent(y, tx, initial_w, max_iters, gamma, grad_function, cost_function):
    """Gradient descent algorithm who work with arbitrary gradient and cost function
    grad and cost function should take y,tw and w as parameter and return resêctivly the gradient vector and the scalar error"""
    # Define parameters to store w and loss
    ws = [initial_w.ravel()]
    losses = []
    w = initial_w.ravel()
    for n_iter in range(max_iters):
        #compute gradient and loss
        gradient=grad_function(y,tx,w)
        loss=cost_function(y,tx,w)
        if n_iter % 50 == 0:
            print("Current iteration={i}, the loss={l}".format(i=n_iter, l=loss))
        #update w by gradient
        w=w-gamma*gradient
        
        # store w and loss
        ws.append(np.copy(w))
        losses.append(loss)
        #print("Gradient Descent({bi}/{ti}): loss={l}, w0={w0}, w1={w1}".format(
        #     bi=n_iter, ti=max_iters - 1, l=loss, w0=w[0], w1=w[1]))

    return losses[-1], ws[-1]


# In[ ]:

def general_stochastic_gradient_descent(y, tx, initial_w, batch_size, max_iters, gamma, stock_grad_function, cost_function):
    """Gradient descent algorithm who work with arbitrary gradient and cost function
    grad and cost function should take y,tw and w as parameter and return respectivly the gradient vector and the scalar error"""
    
    # implement stochastic gradient descent.
    ws = [initial_w.ravel()]
    losses = []
    w = initial_w.ravel()
    
    minibatchs = batch_iter(y, tx, batch_size, num_batches=math.floor(y.shape[0]/batch_size))
    for n_iter in range(0,max_epochs):
        
        # compute gradient and loss
        minibatch=minibatchs.__next__()
        gradient=stock_grad_function(minibatch[0],minibatch[1],w)
        loss=cost_function(y,tx,w)
        
        # update w by gradient
        w=w-gamma*gradient
        # store w and loss
        ws.append(np.copy(w))
        losses.append(loss)
        print("Gradient Descent({bi}/{ti}): loss={l}, w0={w0}, w1={w1}".format(
              bi=n_iter, ti=max_iters - 1, l=loss, w0=w[0], w1=w[1]))

    return losses[-1], ws[-1]


# In[ ]:

def least_squares_GD(y, tx, gamma, max_iters):
    return general_gradient_descent(y,tx,np.zeros((tx.shape[1], 1)).ravel(),max_iters,gamma,compute_gradient_MSE,compute_cost_MSE)

#general_gradient_descent(y,tX,np.zeros((tX.shape[1], 1)),20,0.000003,compute_gradient_MSE,compute_cost_MSE)


# In[ ]:

def least_squares_SGD(y, tx, gamma, max_iters):
    batch_size = y.shape[0]//2
    return general_stochastic_gradient_descent(y,tx,np.zeros((tx.shape[1], 1)),batch_size,max_iters,gamma,compute_gradient_MSE,compute_cost_MSE)


# In[ ]:

def least_squares(y, tx):
    w = np.linalg.solve(tx.T.dot(tx),tx.T.dot(y)) #return best weight
    return compute_cost_MSE(y, tx, w), w


# In[ ]:

def ridge_regression(y, tx, lamb):
    """implement ridge regression."""
    
    lambp = lamb*(2*tx.shape[0])
    #return np.linalg.inv(tx.T.dot(tx)+lambp*np.eye(tx.shape[1])).dot(tx.T).dot(y)
    w =  np.linalg.solve(tx.T.dot(tx)+lambp*np.eye(tx.shape[1]),tx.T.dot(y))
    return compute_cost_MSE(y, tx, w), w


# ### Logistic

# In[ ]:

def sigmoid(t):
    """apply sigmoid function on t."""
    z = np.exp(t)
    return z/(1+z)

def compute_loss_logistic(y, tx, w, lambda_=0):
    """compute the cost by negative log likelihood."""
    clip = np.clip(tx @ w, -700, 700)
    if lambda_ == 0:
        return (1/y.shape[0])*np.sum(np.log(1+np.exp(clip))-y*(tx @ w))
    else:
        return (1/y.shape[0])*np.sum(np.log(1+np.exp(clip))-y*(tx @ w)) + lambda_*np.sum(w*w) #or + lambda_*w.T*w
    #return -np.sum(np.log(1+np.exp(tx @ w))-y*(tx @ w))
    
def compute_gradient_sigmoid(y, tx, w, lambda_=0):
    """compute the gradient of loss."""
    clip = np.clip(tx @ w, -700, 700)
    if lambda_ == 0:
        return (1/y.shape[0])*tx.T.dot(sigmoid(clip)-y)
    else:
        return (1/y.shape[0])*tx.T.dot(sigmoid(clip)-y) + lambda_*2*w


# In[ ]:

def logistic_regression(y, tx, gamma ,max_iters):
    print("fuck")
    ok = general_gradient_descent(y,tx,np.zeros((tx.shape[1], 1)),max_iters,gamma,compute_gradient_sigmoid,compute_loss_logistic)
    return ok


# In[ ]:

def reg_logistic_regression(y, tx, lambda_ , gamma, max_iters):
    gradf = partial(compute_gradient_sigmoid,lambda_ = lambda_)
    costf = partial(compute_loss_logistic, lambda_ = lambda_)
    return general_gradient_descent(y,tx,np.zeros((tx.shape[1], 1)),max_iters,gamma,gradf,costf)


# ### Cross-validation

# In[ ]:

def build_k_indices(y, k_fold, seed):
    """build k indices for k-fold."""
    num_row = y.shape[0]
    interval = int(num_row / k_fold)
    np.random.seed(seed)
    indices = np.random.permutation(num_row)
    k_indices = [indices[k * interval: (k + 1) * interval] for k in range(k_fold)]
    return np.array(k_indices)

def cross_validation(y, tx, function_to_test,k_fold, lambda_,seed,cost_function):
    """return the loss of train values, loss of test values and weights"""
    k_indices=build_k_indices(y,k_fold,seed)
    loss_tr=[]
    loss_te=[]
    weights=[]
    for k in range(k_indices.shape[0]):
        # get k'th subgroup in test, others in train:
        train_indices=k_indices[[i for i in range(len(k_indices)) if i != k]]
        train_tx,train_y=tx[np.ravel(train_indices)],y[np.ravel(train_indices)]
        test_tx,test_y=tx[k_indices[k]],y[k_indices[k]]

        loss_tr_k, weight_k = function_to_test(train_y, train_tx,lambda_)
        loss_tr.append(loss_tr_k)
        weights.append(weight_k)
        loss_te.append(cost_function(test_y,test_tx,weight_k))
    return np.mean(loss_tr), np.mean(loss_te),np.mean(weights,axis=0)

def finding_lambda(y, tx, function_to_test,k_fold,seed,lambdas,cost_function):
    
    # define lists to store the loss of training data and test data
    loss_tr = []
    loss_te = []
    for lambda_ in lambdas:
        print("lambda : ", lambda_)
        if(cost_function==compute_loss_logistic):
            cost_function=partial(compute_loss_logistic,lambda_=lambda_)
        loss_tr_lamb,loss_te_lamb,weight_lamb=cross_validation(y,tx,function_to_test,k_fold,lambda_,seed,cost_function)
        loss_tr.append(loss_tr_lamb)
        loss_te.append(loss_te_lamb)
    return loss_tr,loss_te
        



# ## Preprocessing data formating

# In[ ]:

def remove_outliers(tX,mean_x,std_x):
    tX2=tX.copy()
    for sample in range(tX.shape[0]):
        for dim in range(tX.shape[1]):
            if(tX2[sample,dim]>mean_x[dim]+2*std_x[dim]):
                tX2[sample,dim]=mean_x[dim]
            if(tX2[sample,dim]<mean_x[dim]-2*std_x[dim]):
                tX2[sample,dim]=mean_x[dim]
            if(tX2[sample,dim]==-999):
                tX2[sample,dim]=0
    return tX2
def modify_y(y):
    y2 = y.copy()
    y2[np.where(y==-1)] = 0
    return y2


# In[ ]:

def rescale(tx):
    mins = np.amin(tx,axis = 0)
    maxs = np.amax(tx,axis = 0)
    txscale = (tx-mins)/(maxs-mins)
    return txscale


# ## Generate predictions and save ouput in csv format for submission:

from proj1_helpers import *
DATA_TRAIN_PATH = '../train.csv' # TODO: download train data and supply path here 
y, tX, ids = load_csv_data(DATA_TRAIN_PATH)


# In[ ]:

DATA_TEST_PATH = '../test.csv' # TODO: download train data and supply path here 
_, tX_test, ids_test = load_csv_data(DATA_TEST_PATH)


# In[ ]:

loss,weights = least_squares(y,tX)


# In[ ]:

OUTPUT_PATH = '../submission.csv' # TODO: fill in desired name of output file for submission
y_pred = predict_labels(weights, tX_test)
create_csv_submission(ids_test, y_pred, OUTPUT_PATH)


# In[ ]:



