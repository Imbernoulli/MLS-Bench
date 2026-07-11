"""A STRONG (Mamba/S4D): A = -exp(A_log). Strictly negative, always stable."""
def compute_A(A_log):
    import torch
    return -torch.exp(A_log.float())
