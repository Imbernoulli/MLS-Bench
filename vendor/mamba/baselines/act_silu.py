"""conv-act STRONG (Mamba): SiLU/swish."""
def conv_act(x):
    import torch.nn.functional as F
    return F.silu(x)
