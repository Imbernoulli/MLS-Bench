"""Reference (WEAK): direct scalar yaw regression for mono3d-orientation-encoding."""
import torch.nn as nn
import torch.nn.functional as F


def build_orient_head(emb_dim):
    head = nn.Sequential(nn.Linear(emb_dim, 64), nn.ReLU(), nn.Linear(64, 1))

    def decode(raw):
        return raw[:, 0]

    def loss(raw, yaw_gt):
        return F.smooth_l1_loss(raw[:, 0], yaw_gt)

    return head, decode, loss
