"""Agent-editable surface: PATCH-based training augmentation.

Define `crop_train(img, pts)` -> `(img2, pts2)`, applied to EACH training sample before
its GT density is rendered. `img` is a `(3, H, W)` float32 array in [0,1]; `pts` is a
`(K, 2)` array of `(y, x)` object centres. Return a (possibly cropped) image and the
points that fall inside it, with coordinates adjusted to the crop. The GT count of the
returned sample is `len(pts2)`, so density supervision stays exact. All returned crops
in a run MUST share the same size (they are batched). Only training augmentation
changes; the val split is always the full image.

Training on the FULL image with no augmentation (identity) gives few effective samples
(120 images) -> worse generalisation to the shifted val counts -> higher counting MAE. A
RANDOM-CROP patch augmentation (a random fixed-size crop per image, standard in
MCNN/CSRNet) multiplies effective data and exposes denser local sub-scenes -> lower MAE.

    def crop_train(img, pts):
        import numpy as np
        CROP = 96
        img = np.asarray(img, dtype=np.float32); pts = np.asarray(pts, dtype=np.float32).reshape(-1,2)
        _, H, W = img.shape
        if H <= CROP or W <= CROP: return img, pts
        y0 = np.random.randint(0, H-CROP+1); x0 = np.random.randint(0, W-CROP+1)
        crop = img[:, y0:y0+CROP, x0:x0+CROP].copy()
        if len(pts):
            m = (pts[:,0]>=y0)&(pts[:,0]<y0+CROP)&(pts[:,1]>=x0)&(pts[:,1]<x0+CROP)
            kept = pts[m].copy(); kept[:,0]-=y0; kept[:,1]-=x0
        else:
            kept = pts
        return crop, kept

The DEFAULT below is the deliberately weak IDENTITY (no augmentation). A crashing /
malformed crop_train falls back to the full image.
"""
from __future__ import annotations

import numpy as np


# ================================================================
# EDITABLE REGION — design the patch-based training augmentation below
# ================================================================
def crop_train(img, pts):
    # Default: IDENTITY, no augmentation (weak). Few effective samples -> worse
    # generalisation to the shifted val counts -> higher MAE.
    return (np.asarray(img, dtype=np.float32),
            np.asarray(pts, dtype=np.float32).reshape(-1, 2))
# ================================================================
# END EDITABLE REGION
# ================================================================
