"""Baseline pooling: global AVERAGE pooling (solid default).
Reference: vendor/torchreid-reid/baselines/pool_avg.py
"""
_FILE = "torchreid-reid/solution/pooling.py"
_CONTENT = '''def build_pooling():
    import torch.nn as nn

    pool = nn.AdaptiveAvgPool2d(1)
    pool.name = "avgpool"
    return pool'''
OPS = [
    {"op": "replace", "file": _FILE, "start_line": 12, "end_line": 17, "content": _CONTENT},
]
