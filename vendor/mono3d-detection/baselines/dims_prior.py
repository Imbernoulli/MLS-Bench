"""mono3d-dimension-prior STRONG baseline: residual on the log CLASS-MEAN anchor.

Predict dims as exp(log_mean + 0.3 * residual): start from the log of the mean canonical
dimensions (a statistical shape prior) and let the head predict only a small multiplicative
residual. Because inter-class/intra-class dimension variance is tiny, anchoring on the prior
gives an accurate H immediately, which (via the fixed geometry depth Z=f*H/h2d) also tightens
depth. Lowest dim error and best AP3D. This is the Deep3DBox / MonoDLE dimension recipe.
"""
import torch
import torch.nn as nn


def build_dims_head(emb_dim, log_mean, cls_dims):
    lm = log_mean.detach().clone()

    class _Head(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(nn.Linear(emb_dim, 64), nn.ReLU(), nn.Linear(64, 3))
            self.register_buffer("log_mean", lm)

        def forward(self, emb):
            return self.net(emb)

    head = _Head()

    def decode(raw, ctx):
        return torch.exp(head.log_mean.unsqueeze(0) + 0.3 * raw)

    return head, decode
