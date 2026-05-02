import torch
import torch.nn.functional as F
import math

def scaled_dot_product_attention(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
    """
    Compute scaled dot-product attention.
    """
    raw_attention = torch.matmul(Q, K.transpose(-2, -1))
    scaled_attention = raw_attention / math.sqrt(K.shape[-1])
    return torch.matmul(F.softmax(scaled_attention, dim=-1), V)