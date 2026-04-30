import numpy as np

def pearson_correlation(X):
    """
    Compute Pearson correlation matrix from dataset X.
    """
    X = np.array(X)
    cov_matrix = ((X - np.mean(X, axis=0)).T @ (X - np.mean(X, axis=0))) / (X.shape[0] - 1)
    std = np.std(X, axis=0, ddof=1)
    r = cov_matrix / np.outer(std, std)
    return r