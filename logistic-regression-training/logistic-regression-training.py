import numpy as np

def _sigmoid(z):
    """Numerically stable sigmoid implementation."""
    return np.where(z >= 0, 1/(1+np.exp(-z)), np.exp(z)/(1+np.exp(z)))

def train_logistic_regression(X, y, lr=0.1, steps=1000):
    """
    Train logistic regression via gradient descent.
    Return (w, b).
    """
    w = np.zeros(X.shape[1])
    b = 0
    for _ in range(steps):
        preds = _sigmoid(X @ w + b)

        grad_w = X.T @ (preds - y) / X.shape[1]
        grad_b = np.mean(preds - y)

        w = w - lr * grad_w 
        b = b - lr * grad_b 

    return w, b
    