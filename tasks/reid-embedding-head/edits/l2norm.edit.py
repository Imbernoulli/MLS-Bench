"""Head baseline (pitfall): L2-normalisation before a Euclidean triplet loss.
Reference: vendor/torchreid-reid/baselines/head_l2.py
"""
_FILE = "torchreid-reid/solution/head.py"
_CONTENT = '''def build_head(feat_dim):
    import torch.nn as nn
    import torch.nn.functional as F

    class L2Norm(nn.Module):
        def forward(self, x):
            return F.normalize(x, p=2, dim=1)

    head = L2Norm()
    head.name = "l2norm"
    return head'''
OPS = [
    {"op": "replace", "file": _FILE, "start_line": 12, "end_line": 17, "content": _CONTENT},
]
