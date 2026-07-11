"""Global-mean feature distance.

Negative Euclidean distance to a single GLOBAL feature mean (one isotropic Gaussian, no
per-class structure, no whitening, no normalization). A weak feature score.
Reference: vendor/ood-detection-lab/solution/feature_score.py (pristine).
"""

_FILE = "ood-detection-lab/solution/feature_score.py"

_CONTENT = '''class Scorer:
    def fit(self, ctx):
        self.mu = ctx.tr_feats.double().mean(dim=0)   # [D]
        return self

    def score(self, feats):
        z = feats.double()
        d = (z - self.mu).norm(dim=-1)
        return (-d).numpy()'''

OPS = [
    {"op": "replace", "file": _FILE, "start_line": 13, "end_line": 19, "content": _CONTENT},
]
