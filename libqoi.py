'''
This library contains a collection of utils aimed at working with
Expectations of  Geometric Brownian Motions.
'''
import torch
import numpy as np
import signatory as sg
from math import sqrt
import matplotlib.pyplot as plt

### PART 1: UTILITIES FOR PATHS AND DATASET
def augment(paths):
    '''
    Augment a path givent in input of shape (n_samples, n_times, 1)
    '''
    n_samples = paths.shape[0]
    n_times = paths.shape[1]
    dim = paths.shape[2]
    tmp = torch.zeros(n_samples, n_times, dim + 1)
    assert (dim == 1)

    for nth in range(n_samples):
        tmp[nth][:, 0] = torch.linspace(0., 1., n_times)
        tmp[nth][:, 1] = paths[nth][:, 0]
    return tmp
#---

def simple_bms (n_paths, n_times):
    '''
    Generate n_paths one dimensional Brownian Motion starting at 0
    '''
    w = torch.zeros(n_paths, n_times, 1)
    domain = torch.linspace(0., 1., n_times)
    dt = domain[1]
    for nth in range(1, n_times):
        increments = sqrt(dt) * torch.normal(0., 1., (n_paths, 1))
        w[:, nth] = w[:, nth - 1] + increments
    # Return the simple, not augmented path
    return w
#---

def loggbm_from_bm(bm, mu_par, sigma_par):
    '''
    Compute the log-returns of the geometric Brownian motion derived from
    the Brownian Motion as tensor(n_samples, n_times, 1)
    Note that the tensor in input is NOT augmented.
    '''
    n_samples = bm.shape[0]
    n_times = bm.shape[1]
    dim = bm.shape[2]
    assert(dim == 1)    # I have not yet implemented a general algorithm

    # Tensor containing the result
    log_gbm = torch.zeros(n_samples, n_times, dim)
    # times t, followed by the coefficients for the stoch path
    dt_mesh = torch.linspace(0., 1., n_times)
    right_term = dt_mesh * (mu_par - 0.5*(sigma_par ** 2))
    
    # Compute a log_gbm for every available brownian motion
    for nth in range(n_samples):
        log_gbm[nth][:, 0] = right_term + bm[nth][:, 0]
    
    return log_gbm
#---

def gbm_from_loggbm(log_gbm, s0):
    '''
    Given a log-Geometric Brownian Motion, return the true GBM
    with starting point s0
    '''
    return torch.exp(log_gbm)*s0
#---


### UTILITIES FOR THE DATASET

def mix_and_split(full_x, full_y):
    '''
    Mix the dataset and return train and validation data
    '''
    n_samples = full_x.shape[0]
    half = int(n_samples / 2)
    random_indeces = torch.randperm(n_samples)
    train_indeces = random_indeces[:half]
    val_indeces = random_indeces[half:]
    x_train = full_x[train_indeces]
    y_train = full_y[train_indeces]
    x_val = full_x[val_indeces]
    y_val = full_y[val_indeces]
    return (x_train, y_train, x_val, y_val)
#---


if __name__ == '__main__':
    prova1 = simple_bms(3, 10)
    r = augment(prova1)
    plt.plot(r[0][:, 0], r[0][:, 1])
    plt.show()
