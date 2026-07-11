"""Weak baseline: aggressive bottleneck to a TINY embedding (dim=32) + BNNeck.
Compressing 2048-d ResNet-50 pooled features to 32-d discards discriminative capacity and
hurts retrieval. Reference: vendor/torchreid-reid/baselines/head_dim32.py
"""
_FILE = "torchreid-reid/solution/dimension.py"
_CONTENT = '''def build_embedding_dim(feat_dim):
    return 32'''
OPS = [
    {"op": "replace", "file": _FILE, "start_line": 6, "end_line": 7, "content": _CONTENT},
]
