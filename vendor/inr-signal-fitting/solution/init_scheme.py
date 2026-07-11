"""Agent-editable INR solution surface.

Implement the task declared public callable inside the editable region.
Return a JSON-compatible value satisfying the task contract.
The fixed harness validates the runtime contract and computes reconstruction PSNR.
























"""
from __future__ import annotations

import math

import torch
import torch.nn as nn

import common


# ================================================================
# EDITABLE REGION
# Native starting implementation.
def fit_inr(coords, target, dev):
    # Native starting implementation.
    model = common.SirenMLP(in_dim=2, w0=1.0, w0_hidden=1.0)
    # Native starting implementation.
    with torch.no_grad():
        for m in model.modules():
            if isinstance(m, nn.Linear):
                m.weight.uniform_(-0.05, 0.05)
                if m.bias is not None:
                    m.bias.zero_()
    model = common.train_inr(model, coords, target, dev, label="naive_init")

    @torch.no_grad()
    def predict(coords):
        model.eval()
        return model(coords.to(dev)).detach()

    return predict
# Native starting implementation.
# END EDITABLE REGION
# ================================================================
