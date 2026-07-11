"""Weaker pooling: global MAX pooling (keeps only the peak activation per channel).
Reference: vendor/torchreid-reid/baselines/pool_max.py
"""
_FILE = "torchreid-reid/solution/pooling.py"
_CONTENT = '''def build_pooling():
    import torch.nn as nn

    pool = nn.AdaptiveMaxPool2d(1)
    pool.name = "maxpool"
    return pool'''
OPS = [
    {"op": "replace", "file": _FILE, "start_line": 12, "end_line": 17, "content": _CONTENT},
]
