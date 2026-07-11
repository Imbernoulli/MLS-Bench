"""Deep k-NN on L2-normalized features.

Store the L2-normalized penultimate features of the ID fit set; score = negative distance
to the k-th nearest ID neighbour (Sun et al., ICML 2022). Non-parametric (no Gaussian
assumption). Reference: vendor/ood-detection-lab/baselines/knn.py
"""

_FILE = "ood-detection-lab/solution/feature_score.py"

_CONTENT = '''class Scorer:
    def fit(self, ctx):
        import torch
        eps = 1e-6
        z = ctx.tr_feats.float()
        self.bank = (z / (z.norm(dim=-1, keepdim=True) + eps)).to("cuda")
        self.k = 50; self.eps = eps
        return self

    def score(self, feats):
        import numpy as np, torch
        z = feats.float(); z = z / (z.norm(dim=-1, keepdim=True) + self.eps)
        out = np.empty(z.shape[0], dtype=np.float64); B = 256
        for i in range(0, z.shape[0], B):
            similarity = z[i:i+B].to(self.bank.device) @ self.bank.T
            score = similarity.topk(self.k, dim=1, largest=True).values[:, -1]
            out[i:i+B] = score.cpu().numpy()
        return out'''

OPS = [
    {"op": "replace", "file": _FILE, "start_line": 13, "end_line": 19, "content": _CONTENT},
]
