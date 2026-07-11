"""Agent-editable RGB Jacobian regularization surface.

Implement the task declared public callable inside the editable region.
Return a JSON-compatible value satisfying the task contract.
The fixed harness validates the runtime contract and computes reconstruction PSNR.



















"""
from __future__ import annotations

if False:
    import torch
    import common


# ================================================================
# EDITABLE REGION
def surface_config():
    return {"weight": 1.0}
# END EDITABLE REGION
# ================================================================
