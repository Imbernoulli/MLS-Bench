"""Free-energy logit score.

Energy = T*logsumexp(logits/T) (Liu et al. 2020). Beats MSP on near-OOD by keeping the
logit magnitude, but a purely logit-space score still leaves feature-space structure on
the table. Reference: vendor/ood-detection-lab/baselines/energy.py
"""

_FILE = "ood-detection-lab/solution/score.py"

_CONTENT = '''class Scorer:
    def fit(self, ctx):
        return self

    def score(self, logits, feats):
        import torch
        T = 1.0
        return (T * torch.logsumexp(logits / T, dim=-1)).numpy()'''

OPS = [
    {"op": "replace", "file": _FILE, "start_line": 13, "end_line": 18, "content": _CONTENT},
]
