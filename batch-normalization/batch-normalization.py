import numpy as np

def batch_norm_forward(x, gamma, beta, eps=1e-5):
    """
    Forward-only BatchNorm for (N,D) or (N,C,H,W).
    """
    x = np.array(x)
    gamma = np.array(gamma)
    beta = np.array(beta)
    
    if len(x.shape) == 2:
        m = np.mean(x, axis=0) 
        var = np.var(x, axis=0)
        x_norm = (x - m) / np.sqrt(var + eps)
        return gamma * x_norm + beta
    elif len(x.shape) == 4:
        m = np.mean(x, axis=(0, 2, 3), keepdims=True)
        var = np.var(x, axis=(0, 2, 3), keepdims=True)
        x_norm = (x - m) / np.sqrt(var + eps)
        gamma = gamma.reshape(1, x.shape[1], 1, 1)
        beta = beta.reshape(1, x.shape[1], 1, 1)
        return gamma * x_norm + beta
    else:
        raise ValueError