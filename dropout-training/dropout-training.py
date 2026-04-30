import numpy as np

def dropout(x, p=0.5, rng=None):
    """
    Apply dropout to input x with probability p.
    Return (output, dropout_pattern).
    """
    if rng is None:
        rng = np.random
    x = np.array(x)
    mask = rng.random(size=x.shape)
    pattern = np.where(mask < (1 - p), 1/(1 - p), 0)
    return x * pattern, pattern
    