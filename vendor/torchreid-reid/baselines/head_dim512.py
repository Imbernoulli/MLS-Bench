"""Wide baseline: project the 2048-d ResNet-50 feature to 512 dimensions, then BN.
Reference:
vendor/torchreid-reid/baselines/head_dim512.py

Standalone reference implementation of this baseline (also applied as an edit).
"""


def build_embedding_dim(feat_dim):
    return 512
