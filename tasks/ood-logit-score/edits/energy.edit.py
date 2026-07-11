"""GOOD logit score for ood-logit-score (a strong answer): the free-energy score.

Energy score (Liu et al., NeurIPS 2020): T*logsumexp(logits/T). Unlike MSP this KEEPS the
absolute logit magnitude, so overconfident-but-wrong OOD inputs are separated much better
-> higher AUROC / lower FPR95. Reference: vendor/ood-detection-lab/baselines/energy.py
"""

_FILE = "ood-detection-lab/solution/logit_score.py"

_CONTENT = '''class Scorer:
    def fit(self, ctx):
        return self

    def score(self, logits):
        # Energy score: T*logsumexp(logits/T) (higher = more ID). Keeps logit magnitude.
        import torch
        T = 1.0
        return (T * torch.logsumexp(logits / T, dim=-1)).numpy()'''

OPS = [
    {"op": "replace", "file": _FILE, "start_line": 13, "end_line": 18, "content": _CONTENT},
]
