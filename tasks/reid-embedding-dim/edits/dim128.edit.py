"""Medium baseline: moderate bottleneck to dim=128 + BNNeck.
A 128-d embedding keeps most of the useful capacity. Reference:
vendor/torchreid-reid/baselines/head_dim128.py
"""
_FILE = "torchreid-reid/solution/dimension.py"
_CONTENT = '''def build_embedding_dim(feat_dim):
    return 128'''
OPS = [
    {"op": "replace", "file": _FILE, "start_line": 6, "end_line": 7, "content": _CONTENT},
]
