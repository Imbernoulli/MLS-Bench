"""Agent-editable surface: the MATTING DECODER HEAD.

Return a torch.nn.Module `decoder` whose forward takes the FIXED U-Net encoder's
feature list [e0,e1,e2,e3] (channels [32,64,96,128] at strides [1,2,4,8]) and
outputs a 1-channel alpha matte (B,H,W) in [0,1] at FULL resolution. The encoder,
trimap encoding (raw channel), loss (fixed alpha-L1 + composition), optimiser,
iterations, seed and eval are FIXED; only the decoder changes. Scored by SAD
(LOWER is better) in the trimap UNKNOWN band on a held-out val split.

    def build_decoder(enc_channels):
        import torch, torch.nn as nn, torch.nn.functional as F
        c0, c1, c2, c3 = enc_channels
        class Dec(nn.Module):
            def __init__(self):
                super().__init__()
                def cbr(a,b): return nn.Sequential(nn.Conv2d(a,b,3,padding=1),
                                                   nn.ReLU(True), nn.Conv2d(b,b,3,padding=1),
                                                   nn.ReLU(True))
                self.up3=cbr(c3+c2,c2); self.up2=cbr(c2+c1,c1); self.up1=cbr(c1+c0,c0)
                self.out=nn.Conv2d(c0,1,1)
            def _u(self,x,r): return F.interpolate(x,size=r.shape[-2:],mode="bilinear",align_corners=False)
            def forward(self, feats):
                e0,e1,e2,e3=feats
                d=self.up3(torch.cat([self._u(e3,e2),e2],1))     # SKIP connections
                d=self.up2(torch.cat([self._u(d,e1),e1],1))      # fuse encoder detail
                d=self.up1(torch.cat([self._u(d,e0),e0],1))
                return torch.sigmoid(self.out(d)).squeeze(1)
        return Dec()

The matte's quality lives in the FINE DETAIL along the soft transition (edges).
SKIP CONNECTIONS (U-Net) inject the encoder's high-resolution, low-level features
directly into the decoder so the matte keeps sharp boundaries; guided / index-
guided upsampling (IndexNet, Lu et al. 2019) does likewise. A decoder that only
bilinearly upsamples the DEEPEST (stride-8) feature discards all high-resolution
detail and produces a blurry matte -> high SAD and gradient error.

The DEFAULT below is exactly that weak decoder: it takes ONLY the deepest feature
e3, projects it to alpha, and bilinearly upsamples 8x to full resolution — no skip
connections, no fine detail -> blurry matte, high SAD. Redesigning the decoder with
U-Net skip connections recovers sharp mattes with clear headroom. A malformed /
crashing decoder falls back to the harness U-Net decoder.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


# ================================================================
# EDITABLE REGION — design the matting decoder below
# ================================================================
def build_decoder(enc_channels):
    # Default: deepest-feature-only bilinear-upsample decoder. NO skip connections;
    # discards all high-resolution detail -> blurry matte -> high SAD.
    c0, c1, c2, c3 = enc_channels

    class Dec(nn.Module):
        def __init__(self):
            super().__init__()
            self.proj = nn.Sequential(nn.Conv2d(c3, 32, 3, padding=1), nn.ReLU(True),
                                      nn.Conv2d(32, 1, 1))

        def forward(self, feats):
            e3 = feats[-1]                 # deepest, stride 8 — no skips
            a = self.proj(e3)
            a = F.interpolate(a, size=feats[0].shape[-2:], mode="bilinear",
                              align_corners=False)
            return torch.sigmoid(a).squeeze(1)
    return Dec()
# ================================================================
# END EDITABLE REGION
# ================================================================
