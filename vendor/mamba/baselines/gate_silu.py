"""gate STRONG (Mamba): y * SiLU(z) multiplicative GLU-style gate."""
def gate(y, z):
    import torch.nn.functional as F
    return y * F.silu(z)
