"""mono3d-feature-representation STRONG baseline: FUSE appearance crop + geometry features.

Encode the appearance crop with the CNN AND the geometry feature vector (normalized 2D-box
center/size, log h2d/w2d, aspect, focal) with an MLP, then concatenate and fuse. The geometry
features carry the 2D-box pixel HEIGHT h2d — the key inverse-depth cue for Z=f*H/h2d — and the
box position/aspect, while the crop carries the physical-size and pose signal; fusing both gives
the strongest embedding for the 3D box. This is the standard monocular-3D fusion and matches the
fixed `common.RegionEncoder`. Reference: Deep3DBox / GS3D / MonoDLE all fuse box geometry with
appearance.
"""
import torch
import common


def build_feature_fusion(feat_dim, crop_hw):
    enc = common.RegionEncoder(feat_dim, crop_hw)

    def forward(feat, crop):
        return enc(feat, crop)

    return enc, forward
