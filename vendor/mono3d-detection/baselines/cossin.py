"""Reference (MEDIUM): (cos,sin) regression + atan2 decode for mono3d-orientation-encoding."""
import torch
import torch.nn as nn


def build_orient_head(emb_dim):
    head = nn.Sequential(nn.Linear(emb_dim, 64), nn.ReLU(), nn.Linear(64, 2))

    def decode(raw):
        return torch.atan2(raw[:, 1], raw[:, 0])

    def loss(raw, yaw_gt):
        norm = raw / (raw.norm(dim=1, keepdim=True) + 1e-6)
        tgt = torch.stack([torch.cos(yaw_gt), torch.sin(yaw_gt)], dim=1)
        return ((norm - tgt) ** 2).sum(dim=1).mean()

    return head, decode, loss
