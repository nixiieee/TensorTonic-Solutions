import numpy as np

def global_avg_pool(x):
    """
    Compute global average pooling over spatial dims.
    Supports (C,H,W) => (C,) and (N,C,H,W) => (N,C).
    """
    x = np.array(x)
    if len(x.shape) == 3:
        return np.mean(np.mean(x, axis=2), axis=1)
    elif len(x.shape) == 4:
        return np.mean(np.mean(x, axis=3), axis=2)
    else:
        raise ValueError
    