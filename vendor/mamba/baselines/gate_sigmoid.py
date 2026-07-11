"""gate MID: sigmoid gate y*sigmoid(z) (a gate, but not the SiLU Mamba uses)."""
def gate(y, z):
    import torch
    return y * torch.sigmoid(z)
