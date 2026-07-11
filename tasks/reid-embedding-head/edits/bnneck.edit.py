"""Strong head baseline: BNNeck (BatchNorm1d, bias frozen) -- Luo Bag of Tricks 2019.
Reference: vendor/torchreid-reid/baselines/head_bnneck.py
"""
_FILE = "torchreid-reid/solution/head.py"
_CONTENT = '''def build_head(feat_dim):
    import torch.nn as nn

    class BNNeck(nn.Module):
        def __init__(self, dim):
            super().__init__()
            self.bn = nn.BatchNorm1d(dim)
            self.bn.bias.requires_grad_(False)

        def forward(self, x):
            return self.bn(x)

    head = BNNeck(feat_dim)
    head.name = "bnneck"
    return head'''
OPS = [
    {"op": "replace", "file": _FILE, "start_line": 12, "end_line": 17, "content": _CONTENT},
]
