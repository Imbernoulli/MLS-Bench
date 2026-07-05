# Shadow Removal: Feature Fusion (Last-Block vs Dense Multi-Level Fusion)

## Objective

Design **how the deshadower's decoder features are aggregated** — use only the last decoder
block's features, or **densely fuse** features from every decoder level — so that the
mask-guided residual deshadower removes a cast shadow to match the clean, shadow-free ground
truth as accurately as possible, maximizing **shadow-region PSNR** on a held-out set of
shadowed→clean pairs under a fixed formulation, loss, optimizer, data, and evaluation protocol.

## Background

Shadow removal recovers a shadow-free image from one on which a **cast shadow**
multiplicatively darkens a known, soft-edged region (SP+M-Net linear illumination model,
Le et al., ICCV 2019: `I = a·J`, `a = 1 − (1−att)·m`). The recovery needs BOTH coarse
global-illumination context (to set the umbra brightening level) and fine penumbra-edge detail
(to avoid halos at the soft boundary). A net that reads out only its last decoder block mixes
these into one feature stack; explicitly fusing multiple levels lets each contribute.

- **weak (`fusion=False`)** — read out only the LAST decoder block's features.
- **strong (`fusion=True`)** — DENSE multi-level FEATURE FUSION: concatenate features from every
  decoder level and fuse them with a 1×1 conv (DenseNet / RDN-style), so coarse
  global-illumination and fine penumbra-edge features both feed the output — higher
  shadow-region PSNR.

Everything else — the mask-guided residual formulation, the composite L1+SSIM loss, the base
width/depth, the optimizer, iterations, seed, data, and the metric — is FIXED, so any change in
the score is attributable to the fusion lever.

## Data & pipeline (FIXED)

- Tiny fixed set of REAL shadowed→shadow-free 64×64 photo triplets (+ a real shadow mask) from
  ISTD (Wang et al., CVPR 2018): the same outdoor scene photographed with and without a
  physical cast shadow. THREE severities (**light / medium / heavy**) are terciles of each
  triplet's MEASURED shadow attenuation (mean brightness ratio inside the mask). Train and val
  use ISTD's own **disjoint-scene train/test split**. The shadow-free photo is the ground
  truth, and the shadow **mask is provided as an input**.
- A FIXED mask-guided residual-learning deshadower, a fixed composite L1+SSIM loss up-weighted
  inside the shadow, a fixed optimizer, trained a few hundred steps, fixed seed. Only the
  feature-fusion toggle varies.

## Implementation Contract

Edit **only** `get_fusion_config()` in `image-deshadow/solution/fusion.py`. Return a dict:

```python
def get_fusion_config():
    # {'fusion': True | False}
    return {"fusion": True}
```

A malformed / crashing return falls back to `fusion=False` (weak).

## Metric

The harness prints one line per run:

```
DESHADOW_METRICS surface=fusion setting=<L> psnr=<..> psnr_gain=<..> shadow_psnr=<..> \
    ssim=<..> mse=<..> full_psnr=<..>
```

- **psnr** (PRIMARY, higher better): SHADOW-REGION deshadow PSNR (dB), computed **only over
  pixels the shadow touches**.
- **shadow_psnr**: the shadowed-input copy floor. **psnr_gain** `= psnr − shadow_psnr` must
  be > 0.
- `full_psnr`, `ssim`, `mse` are diagnostics only.

Dense multi-level fusion gives higher shadow-region PSNR than last-block-only, and both clearly
beat the shadowed-input identity floor.
