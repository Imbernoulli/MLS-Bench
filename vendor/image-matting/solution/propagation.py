"""Agent-editable surface: the ALPHA PROPAGATION / GUIDED-FILTER module (propagation).

Return a function
    propagate(alpha, image, trimap) -> refined_alpha (B,H,W) in [0,1]
applied to the decoder's raw alpha as a POST-PROCESSING step. It refines the matte
using IMAGE STRUCTURE (matting affinity): the alpha should follow image edges (the
matting Laplacian, Levin et al. 2008; guided filter, He et al. 2013). It is a
PARAMETER-FREE closed-form filter (no trainable weights). Everything else (data,
network, trimap, loss, optimiser, iterations, seed, eval) is FIXED; only this module
changes. Scored by SAD (LOWER is better) in the trimap UNKNOWN band, gmean over three
trimap-width settings.

Order:
    identity (no propagation)  <  a plain box-smooth (blurs, ignores image edges)
      <  an IMAGE-GUIDED FILTER (edge-aware, aligns the matte to image structure =
         SOTA).

The DEFAULT below is a deliberately weak IDENTITY: no propagation -> the decoder's
raw matte keeps its high-frequency noise and does not snap to image edges -> higher
SAD. Redesign propagate() as an image-GUIDED FILTER (guided filter of `alpha` with
`image` as guidance) that makes the matte edge-aware, with clear headroom. A
malformed / crashing / wrong-shape return falls back to identity.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


# ================================================================
# EDITABLE REGION — design the alpha propagation module below
# ================================================================
def propagate(alpha, image, trimap):
    # Default: IDENTITY (no propagation). The decoder's raw matte keeps its
    # high-frequency noise and is not aligned to image edges -> higher SAD. An
    # image-guided filter (edge-aware smoothing guided by `image`) snaps the matte to
    # image structure and lowers SAD.
    return alpha
# ================================================================
# END EDITABLE REGION
# ================================================================
