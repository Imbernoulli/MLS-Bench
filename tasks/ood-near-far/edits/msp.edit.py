"""Maximum-softmax-probability score.

Maximum Softmax Probability -- weak, and especially poor on NEAR-OOD (CIFAR-100 looks
enough like CIFAR-10 that the softmax stays confident).
Reference: vendor/ood-detection-lab/baselines/msp.py
"""

_FILE = "ood-detection-lab/solution/score.py"

_CONTENT = '''class Scorer:
    def fit(self, ctx):
        return self

    def score(self, logits, feats):
        import torch
        return torch.softmax(logits, dim=-1).max(dim=-1).values.numpy()'''

OPS = [
    {"op": "replace", "file": _FILE, "start_line": 13, "end_line": 18, "content": _CONTENT},
]
