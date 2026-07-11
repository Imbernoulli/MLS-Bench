"""Agent-editable full-resolution training-mask surface for image inpainting."""
from __future__ import annotations

import numpy as np
import torch


# EDITABLE REGION
def make_holes(gt, rng):
    b, _, h, w = gt.shape
    masks = np.zeros((b, 1, h, w), dtype=np.float32)
    for index in range(b):
        side = int(round(float(rng.uniform(0.28, 0.58)) * min(h, w)))
        top = int(rng.integers(0, h - side + 1))
        left = int(rng.integers(0, w - side + 1))
        masks[index, 0, top:top + side, left:left + side] = 1.0
    return torch.from_numpy(masks).to(gt.device)
# END EDITABLE REGION
