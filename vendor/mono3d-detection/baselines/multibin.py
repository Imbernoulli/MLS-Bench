"""Reference (STRONG/SOTA): Deep3DBox MultiBin orientation for mono3d-orientation-encoding."""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def build_orient_head(emb_dim):
    n_bins = 4
    head = nn.Sequential(nn.Linear(emb_dim, 128), nn.ReLU(), nn.Linear(128, n_bins * 3))
    centers = torch.tensor([(-math.pi + 2 * math.pi * (i + 0.5) / n_bins) for i in range(n_bins)])

    def decode(raw):
        B = raw.shape[0]
        logit = raw[:, :n_bins]
        res = raw[:, n_bins:].reshape(B, n_bins, 2)
        b = torch.argmax(logit, dim=1)
        c = centers.to(raw.device)[b]
        r = res[torch.arange(B, device=raw.device), b]
        return c + torch.atan2(r[:, 1], r[:, 0])

    def loss(raw, yaw_gt):
        B = raw.shape[0]
        cen = centers.to(raw.device)
        logit = raw[:, :n_bins]
        res = raw[:, n_bins:].reshape(B, n_bins, 2)
        diff = torch.atan2(torch.sin(yaw_gt.unsqueeze(1) - cen.unsqueeze(0)),
                           torch.cos(yaw_gt.unsqueeze(1) - cen.unsqueeze(0)))
        tgt = torch.argmin(diff.abs(), dim=1)
        ce = F.cross_entropy(logit, tgt)
        delta = diff[torch.arange(B, device=raw.device), tgt]
        r = res[torch.arange(B, device=raw.device), tgt]
        rl = ((r[:, 0] - torch.cos(delta)) ** 2 + (r[:, 1] - torch.sin(delta)) ** 2).mean()
        return ce + rl

    return head, decode, loss
