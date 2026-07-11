"""Head baseline: identity pass-through (raw pooled features).
Reference: vendor/torchreid-reid/baselines/head_raw.py
"""
_FILE = "torchreid-reid/solution/head.py"
_CONTENT = '''def build_head(feat_dim):
    import torch.nn as nn

    head = nn.Identity()
    head.name = "raw"
    return head'''
OPS = [
    {"op": "replace", "file": _FILE, "start_line": 12, "end_line": 17, "content": _CONTENT},
]
