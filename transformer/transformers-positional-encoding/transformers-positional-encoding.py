import numpy as np

def positional_encoding(seq_length: int, d_model: int) -> np.ndarray:
    """
    Generate sinusoidal positional encodings.
    """
    pos = np.arange(seq_length).reshape(-1, 1)
    div_term = np.exp(np.arange(0, d_model, 2) * (-np.log(10000.0) / d_model))
    angles = pos * div_term 
    
    sin_matrix = np.sin(angles)
    cos_matrix = np.cos(angles)

    output_matrix = np.zeros((seq_length, d_model))

    output_matrix[:, 0::2] = sin_matrix
    output_matrix[:, 1::2] = cos_matrix
    return output_matrix
