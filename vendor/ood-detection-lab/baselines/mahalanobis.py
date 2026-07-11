"""Mahalanobis baseline (Lee et al., NeurIPS 2018) with feature normalization.

Fit class-conditional Gaussians on the (L2-normalized) penultimate features with a TIED
covariance. OOD score = negative of the minimum squared Mahalanobis distance to the
class means (higher = closer to a class = more ID). Feature-space; strong on far-OOD.

Feature normalization (z <- z/||z||) is the well-known Mahalanobis++ trick that makes the
tied covariance a much better fit and consistently improves OOD detection.
"""
import numpy as np
import torch

EPS = 1e-6


def _l2norm(x):
    return x / (x.norm(dim=-1, keepdim=True) + EPS)


class Scorer:
    def fit(self, ctx):
        feats = _l2norm(ctx.tr_feats).double()          # [N,D]
        labels = ctx.tr_labels.numpy()
        K = ctx.num_classes
        D = feats.shape[1]
        means = torch.zeros(K, D, dtype=torch.float64)
        # tied covariance: pool centered features across all classes
        centered = torch.empty_like(feats)
        for k in range(K):
            idx = np.where(labels == k)[0]
            mk = feats[idx].mean(dim=0)
            means[k] = mk
            centered[idx] = feats[idx] - mk
        cov = (centered.T @ centered) / feats.shape[0]
        cov += EPS * torch.eye(D, dtype=torch.float64)
        self.means = means
        self.prec = torch.linalg.inv(cov)               # precision matrix
        return self

    def score(self, logits, feats):
        z = _l2norm(feats).double()                     # [N,D]
        # squared Mahalanobis distance to each class mean; take the min over classes
        diff = z.unsqueeze(1) - self.means.unsqueeze(0)  # [N,K,D]
        # (diff @ prec * diff).sum(-1) == mahalanobis^2
        m = torch.einsum("nkd,de,nke->nk", diff, self.prec, diff)  # [N,K]
        min_dist = m.min(dim=1).values                  # [N]
        return (-min_dist).numpy()                       # higher = more ID
