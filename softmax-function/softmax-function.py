import numpy as np

def softmax(x):
    """
    Compute the softmax of input x.
    Works for 1D or 2D NumPy arrays.
    For 2D, compute row-wise softmax.
    """
    x = np.array(x)
    if x.ndim == 2:
        x_shift = x - np.max(x, axis=1, keepdims=True)
        exp_x = np.exp(x_shift)
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)
    return np.exp(x - np.max(x)) / np.sum(np.exp(x - np.max(x)))
        