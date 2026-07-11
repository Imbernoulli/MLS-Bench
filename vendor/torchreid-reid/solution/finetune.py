"""Agent-editable solution surface for person re-identification.

Keep the public callable names and signatures below. Follow the fixed harness
contract for return types, shapes, devices, value ranges, finiteness, and
determinism. The selected implementation is evaluated directly; contract or
runtime failures invalidate the run.
"""
from __future__ import annotations


# EDITABLE REGION
def configure_trainable(backbone):
    for p in backbone.parameters():
        p.requires_grad_(False)
    return None
# END EDITABLE REGION
