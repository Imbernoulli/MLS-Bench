"""conv-act MID: ReLU (a nonlinearity, but hard-saturating / not smooth)."""
def conv_act(x):
    import torch
    return torch.relu(x)
