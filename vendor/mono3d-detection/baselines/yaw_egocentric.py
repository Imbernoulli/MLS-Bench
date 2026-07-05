"""mono3d-yaw-frame WEAK baseline: EGOCENTRIC (global) yaw prediction.

Supervise a MultiBin head on the GLOBAL yaw and decode the global yaw directly, IGNORING the
observation ray. But a monocular crop only reveals the object's LOCAL (ray-relative) orientation:
two objects with identical appearance at different image positions have different global yaw (they
differ by the ray azimuth). Supervising on the global angle forces the head to fit an appearance->
global-yaw map that does not exist, so predictions are systematically rotated for off-center
objects (most of them) -> large yaw error, collapsed BEV/3D IoU. Reference: Deep3DBox / M3D-RPN
note global yaw is NOT a function of appearance; only the allocentric angle is.
"""
import torch.nn as nn
import math
import torch
import torch.nn.functional as F


def _mb(emb_dim):
    n_bins = 4
    head = nn.Sequential(nn.Linear(emb_dim, 128), nn.ReLU(), nn.Linear(128, n_bins * 3))
    centers = torch.tensor([(-math.pi + 2 * math.pi * (i + 0.5) / n_bins) for i in range(n_bins)])
    return head, centers, n_bins


def _decode_angle(raw, centers, n_bins):
    B = raw.shape[0]
    logit = raw[:, :n_bins]
    res = raw[:, n_bins:].reshape(B, n_bins, 2)
    b = torch.argmax(logit, dim=1)
    c = centers.to(raw.device)[b]
    r = res[torch.arange(B, device=raw.device), b]
    return c + torch.atan2(r[:, 1], r[:, 0])


def _mb_loss(raw, ang_gt, centers, n_bins):
    B = raw.shape[0]
    cen = centers.to(raw.device)
    logit = raw[:, :n_bins]
    res = raw[:, n_bins:].reshape(B, n_bins, 2)
    diff = torch.atan2(torch.sin(ang_gt.unsqueeze(1) - cen.unsqueeze(0)),
                       torch.cos(ang_gt.unsqueeze(1) - cen.unsqueeze(0)))
    tgt = torch.argmin(diff.abs(), dim=1)
    ce = F.cross_entropy(logit, tgt)
    delta = diff[torch.arange(B, device=raw.device), tgt]
    r = res[torch.arange(B, device=raw.device), tgt]
    rl = ((r[:, 0] - torch.cos(delta)) ** 2 + (r[:, 1] - torch.sin(delta)) ** 2).mean()
    return ce + rl


def build_yawframe_head(emb_dim):
    head, centers, n_bins = _mb(emb_dim)

    def decode(raw, ctx):
        return _decode_angle(raw, centers, n_bins)          # global yaw directly (no ray)

    def loss(raw, yaw_gt, ctx):
        return _mb_loss(raw, yaw_gt, centers, n_bins)        # supervise on GLOBAL yaw

    return head, decode, loss
