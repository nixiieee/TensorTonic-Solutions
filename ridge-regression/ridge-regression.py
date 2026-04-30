def ridge_regression(X, y, lam):
    """
    Compute ridge regression weights using the closed-form solution.
    """
    X = np.array(X)
    y = np.array(y)
    return np.linalg.inv(X.T @ X + lam * np.identity(X.shape[1])) @ X.T @ y