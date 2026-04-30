import numpy as np

def cross_entropy_loss(y_true, y_pred):
    """
    Compute average cross-entropy loss for multi-class classification.
    """
    y_pred = np.array(y_pred)
    y_true = np.array(y_true) 
    if y_pred.shape[0] != y_true.shape[0]:
        return None

    probs = y_pred[np.arange(y_pred.shape[0]), y_true]
    return -np.mean(np.log(probs))