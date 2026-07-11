"""Baseline pooling: global AVERAGE pooling (solid default).
Reference: vendor/torchreid-reid/baselines/pool_avg.py

Standalone reference implementation of this baseline (also applied as an edit).
"""


def build_pooling():
    import torch.nn as nn

    pool = nn.AdaptiveAvgPool2d(1)
    pool.name = "avgpool"
    return pool
