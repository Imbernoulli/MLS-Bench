"""Good baseline for cv-count-multiscale: MULTI-SCALE context aggregation (CAN-style).

Pools the features at several block sizes (1x1, 2x2, 4x4), upsamples each back and
concatenates them with the original features, giving the density tail explicit
multi-scale CONTEXT. This handles the wide range of object scales / crowding better ->
lower counting MAE. Mirrors CAN (Liu et al., CVPR 2019) / spatial-pyramid pooling.
"""


def build_context(cin):
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    class ContextModule(nn.Module):
        def __init__(self, scales=(2, 4, 8)):
            super().__init__()
            self.scales = scales
            self.projs = nn.ModuleList(
                [nn.Conv2d(cin, cin, 1) for _ in scales])
            self.fuse = nn.Conv2d(cin * (len(scales) + 1), cin, 1)

        def forward(self, x):
            h, w = x.shape[-2:]
            feats = [x]
            for s, proj in zip(self.scales, self.projs):
                p = F.adaptive_avg_pool2d(x, output_size=max(1, min(h, w) // s))
                p = proj(p)
                feats.append(F.interpolate(p, size=(h, w), mode="bilinear",
                                           align_corners=False))
            # RESIDUAL fusion: add multi-scale context back to the base features so the
            # module starts near-identity and converges fast.
            return F.relu(x + self.fuse(torch.cat(feats, dim=1)))

    return ContextModule()
