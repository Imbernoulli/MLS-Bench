"""Weak baseline: aggressive bottleneck to a TINY embedding (dim=32) + BNNeck.
Compressing 2048-d ResNet-50 pooled features to 32-d discards discriminative capacity and
hurts retrieval. Reference: vendor/torchreid-reid/baselines/head_dim32.py

Standalone reference implementation of this baseline (also applied as an edit).
"""


def build_embedding_dim(feat_dim):
    return 32
