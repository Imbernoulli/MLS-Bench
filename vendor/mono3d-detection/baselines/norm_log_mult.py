"""mono3d-depth-normalization STRONG baseline: MULTIPLICATIVE LOG-space residual.

Apply the learned depth correction MULTIPLICATIVELY in log space on top of the geometry depth:
Z = f*H/h2d * exp(0.1 * clamp(residual)). A multiplicative log-space correction is
SCALE-INVARIANT — the same relative correction applies at 6 m and at 40 m — and the exp keeps Z
strictly positive and bounded. This matches the depth range's multiplicative structure, giving
the lowest depth error across all distance regimes. Reference: log/inverse-depth parameterization
is the standard scale-invariant depth normalization (Eigen et al.; GUPNet).
"""
import torch


def build_depth_norm():
    def apply(geom_Z, raw):
        return geom_Z * torch.exp(0.1 * raw[:, 0].clamp(-3, 3))   # scale-invariant, positive

    return apply
