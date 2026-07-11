"""A MID: A = -|A_log| (guaranteed non-positive, but NOT the smooth -exp map;
allows A=0 states with no decay -> weaker than -exp)."""
def compute_A(A_log):
    import torch
    return -torch.abs(A_log.float())
