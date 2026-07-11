"""Medium baseline: moderate bottleneck to dim=128 + BNNeck.
A 128-d embedding keeps most of the useful capacity. Reference:
vendor/torchreid-reid/baselines/head_dim128.py

Standalone reference implementation of this baseline (also applied as an edit).
"""


def build_embedding_dim(feat_dim):
    return 128
