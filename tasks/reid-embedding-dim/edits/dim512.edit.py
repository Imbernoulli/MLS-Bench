"""Wide baseline: project the 2048-d ResNet-50 feature to 512 dimensions, then BN.
Reference:
vendor/torchreid-reid/baselines/head_dim512.py
"""
_FILE = "torchreid-reid/solution/dimension.py"
_CONTENT = '''def build_embedding_dim(feat_dim):
    return 512'''
OPS = [
    {"op": "replace", "file": _FILE, "start_line": 6, "end_line": 7, "content": _CONTENT},
]
