"""Agent-editable surface: MULTI-SCALE CONTEXT aggregation.

Define `build_context(cin)` -> a torch.nn.Module mapping features `(B, cin, h, w)` to
context-enriched features `(B, cin, h, w)`. It runs between the frontend and the density
tail; only the context module changes.

With SINGLE-scale features (identity), objects at scales the frontend's receptive field
does not match are mis-counted -> higher counting MAE. A MULTI-SCALE context module
pools the features at several block sizes (1x1 / 2x2 / 4x4), upsamples each back, and
fuses them with the original features, giving the density tail explicit multi-scale
context -> lower MAE. This mirrors CAN (Liu et al., CVPR 2019) / spatial-pyramid pooling.

    def build_context(cin):
        import torch, torch.nn as nn, torch.nn.functional as F
        class ContextModule(nn.Module):
            def __init__(self, scales=(1,2,4)):
                super().__init__()
                self.scales = scales
                self.projs = nn.ModuleList([nn.Conv2d(cin, cin, 1) for _ in scales])
                self.fuse = nn.Sequential(nn.Conv2d(cin*(len(scales)+1), cin, 1), nn.ReLU(True))
            def forward(self, x):
                h, w = x.shape[-2:]; feats=[x]
                for s, proj in zip(self.scales, self.projs):
                    p = F.adaptive_avg_pool2d(x, output_size=max(1, min(h,w)//s))
                    feats.append(F.interpolate(proj(p), size=(h,w), mode="bilinear", align_corners=False))
                return self.fuse(torch.cat(feats, dim=1))
        return ContextModule()

The DEFAULT below is the deliberately weak SINGLE-scale identity. A crashing / malformed
context module falls back to identity.
"""
from __future__ import annotations

import torch.nn as nn


# ================================================================
# EDITABLE REGION — design the multi-scale context module below
# ================================================================
def build_context(cin):
    # Default: SINGLE-scale (weak). No multi-scale context -> mis-counts off-scale
    # objects -> higher MAE.
    return nn.Identity()
# ================================================================
# END EDITABLE REGION
# ================================================================
