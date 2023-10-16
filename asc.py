'''
LIBRARY DEVELOPMENT
Basic idea of implementing a Signature Average Change to keep track
of how a time series evolve.
'''
import torch
import numpy as np
import matplotlib.pyplot as plt
import signatory as sg

def to_log_returns(ts):
    '''
    Given a database of one dimensional time series, 
    return the time series of log returns
    '''
    assert len(ts.shape) == 2
    # Verify that we work with strict positive numbers
    assert False not in (ts > 0)
#    # Add to the series a very small epsilon to ensure there are no zeroes
#    epsilon = 1e-6
    local_ts = ts # + epsilon
    n_data = local_ts.shape[0]
    len_data = local_ts.shape[1]
    result = torch.zeros(local_ts.shape)
    for nth in range(len_data):
        result[:, nth] = torch.log(local_ts[:,0] / local_ts[:,nth])
    return result
#---


def test_to_log_returns():
    print("Performing three tests")
    ts1 = torch.abs(torch.normal(0., 1., (10, 4)))
    print(to_log_returns(ts1))
    ts2 = torch.zeros(4, 10) + 0.0001
    print(to_log_returns(ts2))
    ts3 = torch.ones(2, 6)
    print(to_log_returns(ts3))
    return 1
#---


def augment_db(ts):
    '''
    Given a database of time series of the form
    (n_data, length_data), return a database of the form
    (n_data, lentgh_data, 2), the added component is the time augmentation.
    '''
    assert len(ts.shape) == 2
    n_data = ts.shape[0]
    len_data = ts.shape[1]
    result = torch.zeros(n_data, len_data, 2)
    for nth in range(n_data):
        result[nth][:, 0] = torch.linspace(0., 1., len_data)
        result[nth][:, 1] = ts[nth]
    return result
#---


def test_augment_db():
    before = torch.normal(0., 1., (3, 1))
    after = augment_db(before)
    print(f"Before: {before}")
    print(f"After Augmentation: {after}")
    before = torch.zeros(2, 4)
    after = augment_db(before)
    print(f"Before: {before}")
    print(f"After Augmentation: {after}")
    before = torch.ones(2, 10)
    after = augment_db(before)
    print(f"Before: {before}")
    print(f"After Augmentation: {after}")
    return 1
#---


def single_ts_to_windowed(ts, window):
    '''
    We take in input a SINGLE time series, so a collection of
    one dimensional samples.
    Return a dataset produced by looking at their windows.
    '''
    assert (window > 0)
    assert len(ts.shape) < 3
    if len(ts.shape) == 1:
        # We have a tensor of shape (len_sample, )
        mydata = ts
    if len(ts.shape) == 2:
        # in this case, be sure we have 1-dimensional data
        if ts.shape[1] == 1:
            mydata = ts.unsqueeze(1)
        elif ts.shape[0] == 1:
            mydata = ts.unsqueeze(0)
        else:
            print("Error. Only 1dim data are supported")
            return 0

    # mydata is my time series of shape (n_samples, )
    len_samples = mydata.shape[0]
    n_samples = len_samples + 1 - window
    assert (n_samples > 0)
    dataset = torch.zeros(n_samples, window)
    for nth in range(n_samples):
        dataset[nth] = ts[nth: nth + window]
    return dataset
#---


def test_window():
    ts1 = torch.tensor(range(1, 11))
    rs1 = single_ts_to_windowed(ts1, 1)
    print(f"Window of 1: {rs1}")
    rs5 = single_ts_to_windowed(ts1, 5)
    print(f"Window of 5: {rs5}")
    rs10 = single_ts_to_windowed(ts1, 10)
    print(f"Window of 10: {rs10}")
    return 1
#---


def relerr(t1, t2, tol = 1e-4):
    '''
    t1 and t2 are two tensors of the same shape, we return their mean error:
    For two reals a, b, the error is |a - b| / max(a, b), and direcly 0
    when a and b are close enough.
    For an array, we do the operation componentwise and take the averages.
    Errors are always in PERCENTAGE.
    '''
    # First, linearize the tensors
    t1 = t1.reshape(-1)
    t2 = t2.reshape(-1)
    assert (len(t1) == len(t2))
    valid_indeces = []
    # Ignore the indeces where both t1 and t2 have enough similar elements,
    # essentially to avoid division by zero
    # When taking the final mean error, an equal number of zeros will
    # be plugged int.
    for i in range(len(t1)):
        if torch.abs(t1[i] - t2[i]) > tol:
            valid_indeces.append(i)
    if len(valid_indeces) > 0:
        zeros_toadd = len(t1) - len(valid_indeces)
        x = t1[valid_indeces]
        y = t2[valid_indeces]
        # Do not foget to condier the absolute values BEFORE the max!!!
        denom = torch.max(torch.abs(x), torch.abs(y))
        coordinate_rerrs = torch.abs((x - y) / denom) * 100.
        complete_errors=torch.cat((coordinate_rerrs,torch.zeros(zeros_toadd)))
        tmp = torch.mean(complete_errors, dim = 0).item()
        result = tmp
    else:
        result = 0.
    return result
#---


def consecutive_errs(db, tols):
    '''
    Given a tensor of shape (n_samples, len_samples),
    compute the differences between subsequent samples.
    '''
    assert (len(db.shape) == 2)
    n_samples = db.shape[0]
    assert (n_samples > 1)
    result = torch.zeros(n_samples - 1)
    for nth in range(n_samples - 1):
        result[nth] = relerr(db[nth], db[nth+1], tols)
    return result
#---


def test_consecutive_errs():
    db1 = torch.ones(10, 1)
    res1 = consecutive_errs(db1)
    print(f"On constant 1-data: {res1}")
    db2 = torch.ones(10, 4)
    res2 = consecutive_errs(db2)
    print(f"On constant 4-data: {res2}")
    return 1
#---


def asc(ts, window, sig_depth = 5, tols = 1e-3):
    '''
    Given a one dimensional time series on which we observe a certain
    window, return the Average-Signature-Change.
    '''
    # Check that we simply have a collection of numbers
    assert len(ts.shape) == 1
    # Produce a database composed by the "windowed" time series
    db0 = single_ts_to_windowed(ts, window)
    # Convert the data into their log-returns, so to start from 0
    db1 = to_log_returns(db0)
    # Extend them by adding the time coordinate
    db2 = augment_db(db1)
    # Compute their signature
    db3 = sg.signature(db2, depth = sig_depth)
    # Compute the consecutive differences between the signatures
    result = consecutive_errs(db3, tols)
    return result
#---

'''
Let's perform now a complete simulation!
'''
len_ts = 200
to_show = 100

# Generate a random time series of positive numbers
ts = torch.abs(torch.normal(0., 4., (len_ts,)))

# Longer and shorter time windows
slow_window = 15
fast_window = 5
assert (slow_window > fast_window)

# Their respective AverageSignatureChanges
slow_asc = asc(ts, slow_window)
fast_asc = asc(ts, fast_window)

# Take just the last to_view points to visualize
d1 = slow_asc[:-to_show]
d2 = fast_asc[:-to_show]
d3 = ts[:-to_show]

# Plot the results
fig, (ax0, ax1) = plt.subplots(2)
ax0.plot(d3, label="stock price", color="teal")
ax1.plot(d1, label="slow-asc", color='green')
ax1.plot(d2, label='fast-asc', color='red')
ax0.grid()
ax1.grid()
ax0.legend()
ax1.legend()
plt.show()
