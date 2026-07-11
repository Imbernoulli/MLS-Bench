"""Class-conditional Mahalanobis score with a tied covariance.

Fit class-conditional Gaussians on L2-NORMALIZED penultimate features with a TIED
covariance; score = negative min squared Mahalanobis distance to the class means. The L2
normalization (Mahalanobis++) makes the tied covariance a much better fit and sharply
improves OOD detection. Reference: vendor/ood-detection-lab/baselines/mahalanobis.py
"""

_FILE = "ood-detection-lab/solution/feature_score.py"

_CONTENT = '''class Scorer:
    def fit(self, ctx):
        import numpy as np, torch
        eps = 1e-6
        feats = ctx.tr_feats.double()
        feats = feats / (feats.norm(dim=-1, keepdim=True) + eps)   # L2 normalize
        labels = ctx.tr_labels.numpy(); K = ctx.num_classes; D = feats.shape[1]
        means = torch.zeros(K, D, dtype=torch.float64)
        centered = torch.empty_like(feats)
        for k in range(K):
            idx = np.where(labels == k)[0]
            mk = feats[idx].mean(dim=0); means[k] = mk
            centered[idx] = feats[idx] - mk
        cov = (centered.T @ centered) / feats.shape[0] + eps * torch.eye(D, dtype=torch.float64)
        self.means = means; self.prec = torch.linalg.inv(cov); self.eps = eps
        return self

    def score(self, feats):
        import torch
        z = feats.double(); z = z / (z.norm(dim=-1, keepdim=True) + self.eps)
        diff = z.unsqueeze(1) - self.means.unsqueeze(0)            # [N,K,D]
        m = torch.einsum("nkd,de,nke->nk", diff, self.prec, diff)  # maha^2
        return (-m.min(dim=1).values).numpy()'''

OPS = [
    {"op": "replace", "file": _FILE, "start_line": 13, "end_line": 19, "content": _CONTENT},
]
