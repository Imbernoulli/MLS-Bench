"""Monocular-3D DEPTH-NORMALIZATION design surface (agent-editable) for mono3d-depth-normalization.

The geometry depth Z0 = f*H/h2d is corrected by a learned residual from the head. The QUESTION is
HOW to parameterize that correction. Metric depth spans 6-40 m, a multiplicative range.

You implement:

    def build_depth_norm() -> callable:

`apply(geom_Z, raw) -> Z [B]` combines the analytic geometry depth `geom_Z` [B] with the head's
raw output `raw` [B,k] into the final POSITIVE metric depth.

The DEFAULT below is the WEAK baseline: a RAW ADDITIVE residual in metres (Z = geom_Z + raw[:,0]).
A single additive metre-scale term is badly scaled across the 6-40 m range and can push Z negative.
A MULTIPLICATIVE LOG-space residual (Z = geom_Z * exp(0.1*clamp(raw))) is scale-invariant and
strictly positive, matching the depth range's multiplicative structure. Everything else is fixed.
"""
from __future__ import annotations

import torch


# ================================================================
# EDITABLE REGION — build apply(geom_Z, raw) -> Z
# ================================================================
def build_depth_norm():
    # WEAK DEFAULT: RAW ADDITIVE residual in metres (badly scaled, can go <= 0).
    def apply(geom_Z, raw):
        return geom_Z + raw[:, 0]

    return apply
# ================================================================
# END EDITABLE REGION
# ================================================================
