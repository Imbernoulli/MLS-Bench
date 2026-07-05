"""Agent-editable surface: the OUTPUT-STRIDE / UPSAMPLING decoder.

Define `build_decoder(cin)` -> a torch.nn.Module mapping the stride-8 features
`(B, cin, h, w)` to refined features `(B, C, h', w')` that feed the density tail. The
count is the integral of the resulting density map (resolution-invariant: the harness
rescales the integral to keep the count fair regardless of output resolution).

A COARSE stride-8 map (identity decoder) cannot separate objects that fall in the same
16x16 cell in dense scenes -> the count saturates -> higher counting MAE. A learned
UPSAMPLING decoder (transposed conv to stride 4 + refinement) produces a finer,
higher-quality density map so nearby objects occupy separate cells -> lower MAE. Finer
output resolution is the lever behind TEDnet / SANet decoders.

    def build_decoder(cin):
        import torch.nn as nn
        class UpDecoder(nn.Module):
            def __init__(self):
                super().__init__()
                self.up = nn.ConvTranspose2d(cin, cin, 2, stride=2)
                self.refine = nn.Sequential(nn.Conv2d(cin, cin, 3, padding=1), nn.ReLU(True))
            def forward(self, x): return self.refine(self.up(x))
        return UpDecoder()

The DEFAULT below is the deliberately weak IDENTITY decoder (coarse stride-8 output). A
crashing / malformed decoder falls back to identity.
"""
from __future__ import annotations

import torch.nn as nn


# ================================================================
# EDITABLE REGION — design the output-stride / upsampling decoder below
# ================================================================
def build_decoder(cin):
    # Default: IDENTITY decoder (weak). Coarse stride-8 map -> objects in one cell
    # cannot be separated in dense scenes -> higher MAE.
    return nn.Identity()
# ================================================================
# END EDITABLE REGION
# ================================================================
