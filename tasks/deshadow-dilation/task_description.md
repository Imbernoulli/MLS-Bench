# Shadow Removal: Receptive Field / Dilation (Small-RF vs Dilated Deshadower)

## Objective

Design the **dilation schedule** of the mask-guided residual deshadower's bottleneck blocks —
plain 3×3 convolutions (small receptive field) or **dilated** convolutions (large receptive
field) — so that it removes a cast shadow to match the clean, shadow-free ground truth as
accurately as possible, maximizing **shadow-region PSNR** on a held-out set of shadowed→clean
pairs under a fixed formulation, loss, optimizer, data, and evaluation protocol.

## Background

Shadow removal recovers a shadow-free image from one on which a **cast shadow**
multiplicatively darkens a known, soft-edged region. Under the physics-based linear
illumination model of Shadow Image Decomposition / SP+M-Net (Le et al., ICCV 2019) the
shadowed image is `I = a·J` with `a(x) = 1 − (1−att)·m(x)`, where `J` is the clean scene,
`m` is the soft shadow matte/mask and `att` the per-image umbra attenuation. The recovery is
a spatially-varying brightening, and the net must see the **whole extent** of a shadow to
brighten its umbra centre consistently. A small receptive field only sees a local patch, so
it under-corrects the middle of a large soft shadow.

- **weak (dilations `[1, 1]`)** — plain 3×3 bottleneck convolutions, a SMALL receptive field.
  The net cannot see the full extent of a large soft shadow in one pass, so it under-corrects
  the umbra centre — lower shadow-region PSNR.
- **strong (dilations `[2, 4]`)** — DILATED bottleneck convolutions (à la ASPP / multi-context
  deshadowing), a much LARGER receptive field that covers big shadows and models the smooth
  penumbra falloff — higher shadow-region PSNR. Helps most on the larger, darker **heavy**
  shadows.

Everything else — the mask-guided residual formulation, the composite L1+SSIM loss, the base
width/depth, the optimizer, iterations, seed, data, and the metric — is FIXED, so any change
in the score is attributable to the dilation schedule.

## Data & pipeline (FIXED)

- Tiny fixed set of REAL shadowed→shadow-free 64×64 photo triplets (+ a real shadow mask) from
  ISTD (Wang et al., CVPR 2018): the same outdoor scene photographed with and without a
  physical cast shadow. THREE severities (**light / medium / heavy**) are terciles of each
  triplet's MEASURED shadow attenuation (mean brightness ratio inside the mask). Train and val
  use ISTD's own **disjoint-scene train/test split**. The shadow-free photo is the ground
  truth. The shadow **mask is provided as an input**.
- A FIXED mask-guided residual-learning deshadower (`clean = shadowed + net([shadowed, mask])`),
  a fixed composite loss (L1 + SSIM, up-weighted inside the shadow), a fixed optimizer,
  trained a few hundred steps, fixed seed. Only the dilation schedule varies.

## Implementation Contract

Edit **only** `get_dilation_config()` in `image-deshadow/solution/dilation.py`. Return a dict:

```python
def get_dilation_config():
    # {'dilations': [d1, d2]}  (dilation rate of the two bottleneck conv blocks)
    return {"dilations": [2, 4]}
```

A malformed / crashing return falls back to `dilations=[1, 1]` (weak).

## Metric

The harness prints one line per run:

```
DESHADOW_METRICS surface=dilation setting=<L> psnr=<..> psnr_gain=<..> shadow_psnr=<..> \
    ssim=<..> mse=<..> full_psnr=<..>
```

- **psnr** (PRIMARY, higher better): SHADOW-REGION deshadow PSNR (dB) of the deshadowed output
  vs the clean GT, computed **only over pixels the shadow touches** — a method that merely
  copies the LIT region gains nothing; it must actually brighten the shadow.
- **shadow_psnr**: PSNR of the shadowed input over the shadow region — the copy / do-nothing
  floor. **psnr_gain** `= psnr − shadow_psnr` must be > 0.
- `full_psnr`, `ssim`, `mse` are diagnostics only.

The dilated trunk gives higher shadow-region PSNR than the small-RF trunk (most on the larger
heavy shadows), and both clearly beat the shadowed-input identity floor.
