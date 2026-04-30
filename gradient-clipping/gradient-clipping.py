import numpy as np

def clip_gradients(g, max_norm):
    """
    Clip gradients using global norm clipping.
    """
    g = np.array(g)
    g_norm = np.linalg.norm(g)
    g_clip = np.where(g_norm <= max_norm or max_norm <= 0 or g_norm <= 0, g, g * (max_norm / g_norm))
    return g_clip