"""Agent-editable trimap-encoding surface for full-inventory image matting.

Keep encode_trimap(trimap). Return one to eight finite feature planes with the same
batch and spatial dimensions. The selected function is executed directly.
"""
from __future__ import annotations

import torch


# EDITABLE REGION
def encode_trimap(trimap):
    # Native raw-value encoding.
    return trimap.unsqueeze(1)
# END EDITABLE REGION
