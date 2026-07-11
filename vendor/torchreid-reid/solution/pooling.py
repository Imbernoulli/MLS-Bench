"""Agent-editable solution surface for person re-identification.

Keep the public callable names and signatures below. Follow the fixed harness
contract for return types, shapes, devices, value ranges, finiteness, and
determinism. The selected implementation is evaluated directly; contract or
runtime failures invalidate the run.
"""
from __future__ import annotations


# EDITABLE REGION
def build_pooling():
    import torch.nn as nn

    pool = nn.AdaptiveAvgPool2d(1)
    pool.name = "avgpool"
    return pool
# END EDITABLE REGION

