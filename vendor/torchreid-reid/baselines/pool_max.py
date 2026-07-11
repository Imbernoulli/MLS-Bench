"""Weaker pooling: global MAX pooling (keeps only the peak activation per channel).
Reference: vendor/torchreid-reid/baselines/pool_max.py

Standalone reference implementation of this baseline (also applied as an edit).
"""


def build_pooling():
    import torch.nn as nn

    pool = nn.AdaptiveMaxPool2d(1)
    pool.name = "maxpool"
    return pool
