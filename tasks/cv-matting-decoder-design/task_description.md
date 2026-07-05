# Image Matting: Design the Decoder (bilinear upsample vs U-Net skip connections)

## Research Question
Trimap-guided **image matting** predicts a **soft foreground alpha** `α ∈ [0,1]` per
pixel (NOT a hard segmentation mask) via `I = α·F + (1−α)·B`. The matte's quality lives
in the **fine detail along the soft transition** — thin structures, hair, edges. The
**decoder** that maps the encoder's features back to a full-resolution alpha map
determines whether that detail survives. A decoder that only bilinearly upsamples the
**deepest (low-resolution) feature** discards all high-resolution detail and produces a
blurry matte. **Skip connections** (U-Net) inject the encoder's high-resolution,
low-level features directly into the decoder so the matte keeps sharp boundaries;
index-guided upsampling (IndexNet, Lu et al. 2019) does likewise. **Redesign the
decoder** to recover fine matte detail.

The fixed harness handles the data, the encoder, the trimap conditioning, the loss
(fixed alpha-L1 + composition), the optimiser, the iterations, the seed, and the
evaluation; your contribution is the **decoder head only**.

## Background
The standard matting metric is the **alpha SAD** (`/1000`) computed **only in the
trimap unknown band** (lower is better); the **gradient error** (which rewards sharp
edges) and MSE are secondary. Because the ground-truth alpha in the unknown band is a
genuine **soft ramp** with fine high-frequency structure (mean ≠ 0.5), a
**constant-0.5 predictor scores `CONST_HALF_SAD`** (far above any real net), so the
metric is monotone in matting quality. A **deepest-feature bilinear** decoder cannot
reproduce the fine detail and scores worse than a U-Net skip-connection decoder.

## Implementation Contract
Modify `build_decoder` in `image-matting/solution/decoder.py` to return a
`torch.nn.Module` whose forward takes the fixed encoder's feature list
`[e0,e1,e2,e3]` (channels `[32,64,96,128]` at strides `[1,2,4,8]`) and outputs a
`(B,H,W)` alpha map in `[0,1]` at full resolution:

```python
def build_decoder(enc_channels):
    import torch, torch.nn as nn, torch.nn.functional as F
    c0, c1, c2, c3 = enc_channels
    def cbr(a, b):
        return nn.Sequential(nn.Conv2d(a, b, 3, padding=1), nn.ReLU(True),
                             nn.Conv2d(b, b, 3, padding=1), nn.ReLU(True))
    class Dec(nn.Module):
        def __init__(self):
            super().__init__()
            self.up3 = cbr(c3+c2, c2); self.up2 = cbr(c2+c1, c1); self.up1 = cbr(c1+c0, c0)
            self.out = nn.Conv2d(c0, 1, 1)
        def _u(self, x, r):
            return F.interpolate(x, size=r.shape[-2:], mode="bilinear", align_corners=False)
        def forward(self, feats):
            e0, e1, e2, e3 = feats
            d = self.up3(torch.cat([self._u(e3, e2), e2], 1))    # SKIP connections
            d = self.up2(torch.cat([self._u(d, e1), e1], 1))
            d = self.up1(torch.cat([self._u(d, e0), e0], 1))
            return torch.sigmoid(self.out(d)).squeeze(1)
    return Dec()
```

- Return a `torch.nn.Module`. A malformed / crashing decoder falls back to the harness
  U-Net decoder.

## Fixed Pipeline & Evaluation
- Data: 100 train / 40 val **synthetic composites** (128×128), `I = α·F + (1−α)·B` with
  an **exact** soft GT alpha (blobby shape with fine hair-like detail) and a derived
  trimap.
- Network: a **fixed U-Net encoder** + your decoder, fed RGB + the trimap, trained a
  short fine-tune with a fixed alpha-L1 + composition loss. **Only the decoder changes.**
- Settings: three **trimap-width** difficulties — `medium` (band width 6), `wide`
  (band width 9) and `xwide` (band width 12, thickest unknown band, hardest). The
  trimap is re-derived from the exact GT alpha at eval time; training uses the medium
  band. The score is the **gmean over all three settings** (`wide`/`xwide` hidden).
- Metric (lower is better): **alpha SAD** in the trimap unknown band on the val split;
  MSE and gradient error are also recorded.
- The scoring midpoint sits between the bilinear start and a U-Net skip-connection
  decoder: you score above 0.5 only by redesigning the decoder with headroom.
