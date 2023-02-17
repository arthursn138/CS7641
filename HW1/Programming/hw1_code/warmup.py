import numpy as np

def indices_of_k(arr, k):
    M = np.where(arr == k)[0]
    return M

def argmax_1d(arr):
    carlos = np.argmax(arr)
    return carlos

def mean_rows(arr):
    zoe = np.mean(arr, axis = 1)
    return zoe

def sum_squares(arr):
    sumsq = np.sum(np.square(arr), axis = 1, keepdims=True)
    return sumsq

def fast_manhattan(x, y):
    fast = np.sum(np.abs(x[:,np.newaxis]-y), axis=-1)
    return fast


def multiple_choice():
    choice = 1
    return choice
