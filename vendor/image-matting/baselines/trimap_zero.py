"""Weak baseline (negative control) for cv-matting-trimap-encoding: no trimap.

An all-zeros plane throws away the trimap -> the net is trimap-blind and must infer
the matte from RGB colour alone -> high SAD in the unknown band. This is the starting
default in vendor/image-matting/solution/trimap.py.
"""


def encode_trimap(trimap):
    import torch
    return torch.zeros_like(trimap).unsqueeze(1)
