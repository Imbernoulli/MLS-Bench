"""Agent-editable INR solution surface.

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
    return {"n_levels": 1, "base_res": 4, "finest_res": 4}
# END EDITABLE REGION
# ================================================================
