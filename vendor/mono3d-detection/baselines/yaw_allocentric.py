"""mono3d-yaw-frame STRONG baseline: ALLOCENTRIC (ray-relative) yaw + ray composition.

Supervise the MultiBin head on the ALLOCENTRIC angle (global yaw minus the observation-ray
azimuth) and recover the global (egocentric) yaw at decode by adding the ray back:
    global_yaw = allocentric_yaw + atan2(x, z).
The allocentric angle is exactly what a monocular crop determines (invariant to WHERE in the
image the object appears), so the head learns a well-posed appearance->angle map and the ray
composition supplies the missing global rotation analytically. Far lower yaw error and higher
3D IoU. This is the Deep3DBox / M3D-RPN local-vs-global orientation convention.
"""
import torch.nn as nn
import math
import torch
import torch.nn.functional as F


def build_yawframe_head(emb_dim):
    n_bins = 4
    head = nn.Sequential(nn.Linear(emb_dim, 128), nn.ReLU(), nn.Linear(128, n_bins * 3))
    centers = torch.tensor([(-math.pi + 2 * math.pi * (i + 0.5) / n_bins) for i in range(n_bins)])

    def _decode_angle(raw):
        B = raw.shape[0]
        logit = raw[:, :n_bins]
        res = raw[:, n_bins:].reshape(B, n_bins, 2)
        b = torch.argmax(logit, dim=1)
        c = centers.to(raw.device)[b]
        r = res[torch.arange(B, device=raw.device), b]
        return c + torch.atan2(r[:, 1], r[:, 0])

    def decode(raw, ctx):
        return _decode_angle(raw) + ctx["ray"].reshape(-1)   # add observation-ray azimuth back

    def loss(raw, yaw_gt, ctx):
        alloc_gt = yaw_gt - ctx["ray"].reshape(-1)           # supervise on ALLOCENTRIC target
        B = raw.shape[0]
        cen = centers.to(raw.device)
        logit = raw[:, :n_bins]
        res = raw[:, n_bins:].reshape(B, n_bins, 2)
        diff = torch.atan2(torch.sin(alloc_gt.unsqueeze(1) - cen.unsqueeze(0)),
                           torch.cos(alloc_gt.unsqueeze(1) - cen.unsqueeze(0)))
        tgt = torch.argmin(diff.abs(), dim=1)
        ce = F.cross_entropy(logit, tgt)
        delta = diff[torch.arange(B, device=raw.device), tgt]
        r = res[torch.arange(B, device=raw.device), tgt]
        rl = ((r[:, 0] - torch.cos(delta)) ** 2 + (r[:, 1] - torch.sin(delta)) ** 2).mean()
        return ce + rl

    return head, decode, loss
