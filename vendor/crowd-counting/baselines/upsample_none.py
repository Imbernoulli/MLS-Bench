"""Weak baseline for cv-count-upsample: NO decoder (coarse stride-8 output).

Identity decoder: the density map stays at stride 8 (16x16 for a 128px image). In
dense scenes many objects fall inside a single coarse cell and cannot be separated, so
the count saturates -> higher counting MAE in crowded regions.
"""


def build_decoder(cin):
    import torch.nn as nn
    return nn.Identity()
