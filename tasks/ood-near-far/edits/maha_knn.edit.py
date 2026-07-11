"""A hybrid of energy (logits) and deep k-NN (features).

Combine the free-energy logit score with the deep k-NN feature-space score (negative
distance to the k-th nearest ID neighbour on L2-normalized penultimate features), each
standardized on the ID fit set and summed. Using BOTH the logit magnitude AND the
feature-space density handles the near-OOD (CIFAR-100) and far-OOD (SVHN) regimes better
than either alone. Reference: baselines/energy.py + baselines/knn.py
"""

_FILE = "ood-detection-lab/solution/score.py"

_CONTENT = '''class Scorer:
    def fit(self, ctx):
        import torch
        eps = 1e-6
        z = ctx.tr_feats.float()
        self.bank = (z / (z.norm(dim=-1, keepdim=True) + eps)).to("cuda")
        self.k = 50; self.eps = eps
        # standardization stats for the two component scores on the ID fit set
        e_tr = torch.logsumexp(ctx.tr_logits.double(), dim=-1)
        m_tr = self._knn(ctx.tr_feats, neighbors=51)
        self.e_mu, self.e_sd = float(e_tr.mean()), float(e_tr.std() + eps)
        self.m_mu, self.m_sd = float(m_tr.mean()), float(m_tr.std() + eps)
        return self

    def _knn(self, feats, neighbors=50):
        import numpy as np, torch
        z = feats.float(); z = z / (z.norm(dim=-1, keepdim=True) + self.eps)
        out = torch.empty(z.shape[0], dtype=torch.float64); B = 256
        for i in range(0, z.shape[0], B):
            similarity = z[i:i+B].to(self.bank.device) @ self.bank.T
            score = similarity.topk(neighbors, dim=1, largest=True).values[:, -1]
            out[i:i+B] = score.double().cpu()
        return out                       # higher = more ID

    def score(self, logits, feats):
        import torch
        e = torch.logsumexp(logits.double(), dim=-1)
        m = self._knn(feats)
        s = (e - self.e_mu) / self.e_sd + (m - self.m_mu) / self.m_sd
        return s.numpy()'''

OPS = [
    {"op": "replace", "file": _FILE, "start_line": 13, "end_line": 18, "content": _CONTENT},
]
