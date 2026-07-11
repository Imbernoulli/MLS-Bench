"""KNN baseline (Sun et al., ICML 2022): deep nearest-neighbor OOD detection.

Store the L2-normalized penultimate features of the ID fit set. For a test point, the OOD
score = negative distance to its k-th nearest ID neighbour (higher = closer to the ID
manifold = more ID). Non-parametric (no Gaussian assumption); very strong on far-OOD.
Feature normalization is essential.
"""
import numpy as np
import torch

EPS = 1e-6
K_NN = 50


def _l2norm(x):
    return x / (x.norm(dim=-1, keepdim=True) + EPS)


class Scorer:
    def fit(self, ctx):
        self.bank = _l2norm(ctx.tr_feats).float()       # [N,D]
        self.k = K_NN
        return self

    def score(self, logits, feats):
        z = _l2norm(feats).float()                      # [M,D]
        # euclidean distance to every train embedding, in chunks to bound memory
        out = np.empty(z.shape[0], dtype=np.float64)
        B = 512
        for i in range(0, z.shape[0], B):
            zb = z[i:i + B]
            d = torch.cdist(zb, self.bank)              # [b,N]
            kth = d.kthvalue(self.k, dim=1).values      # k-th nearest distance
            out[i:i + B] = (-kth).numpy()               # higher = more ID
        return out
