# Shadow Removal: Multi-Scale (Single-Scale vs Coarse-to-Fine Pyramid)

## Objective

Design **whether the mask-guided deshadower is single-scale or a coarse-to-fine pyramid** —
so that it removes a cast shadow to match the clean, shadow-free ground truth as accurately as
possible, maximizing **shadow-region PSNR** on a held-out set of shadowed→clean pairs under a
fixed formulation, loss, optimizer, data, and evaluation protocol.

## Background

Shadow removal recovers a shadow-free image from one on which a **cast shadow**
multiplicatively darkens a known, soft-edged region (SP+M-Net linear illumination model,
Le et al., ICCV 2019: `I = a·J`, `a = 1 − (1−att)·m`). Large, dark shadows span a big spatial
extent, so a single-scale net at full resolution struggles to relight the umbra centre
consistently; processing a **coarse** version first gives a large effective receptive field
that captures the whole shadow, and the coarse estimate then guides a full-resolution refine.

- **weak (`multiscale=False`)** — a single-scale mask-guided U-Net.
- **strong (`multiscale=True`)** — a COARSE-TO-FINE pyramid: relight at HALF resolution first
  (large effective receptive field, captures the whole big soft shadow) then fuse that coarse
  estimate into a full-resolution refinement branch (MSPFN / pyramid restoration style).

## Honesty note

On this synthetic near-affine multiplicative shadow the pyramid's benefit is **concentrated in
the larger shadows**: it helps clearly on the **heavy** setting (big, dark umbrae) and on
**light**, but on the **medium** setting it is a near-tie / marginally behind the single-scale
net (the medium shadow already fits in the single-scale receptive field). The TASK-LEVEL
geometric mean of shadow-region PSNR is monotone strong > weak across the validated seeds, so
the surface is a valid weak→strong lever at the score level; it is a **weaker** lever than the
mask-guidance surface. Anchors use a per-setting floor just below the weaker of the two
references so the curve is well-formed on every setting, and the light/heavy gains dominate the
aggregate.

## Data & pipeline (FIXED)

- Tiny fixed set of shadowed→clean 64×64 pairs (+ a soft shadow mask) at THREE cast-shadow
  severities (**light / medium / heavy**). Train and val use **disjoint patches AND disjoint
  shadow realisations**. The clean patch is the **exact** ground truth. The soft shadow **mask
  is provided as an input**.
- A FIXED mask-guided residual-learning deshadower, a fixed composite L1+SSIM loss up-weighted
  inside the shadow, a fixed optimizer, trained a few hundred steps, fixed seed. Only the
  single-scale-vs-pyramid toggle varies (same per-branch width).

## Implementation Contract

Edit **only** `get_multiscale_config()` in `image-deshadow/solution/multiscale.py`. Return:

```python
def get_multiscale_config():
    # {'multiscale': True | False}
    return {"multiscale": True}
```

A malformed / crashing return falls back to `multiscale=False` (weak).

## Metric

```
DESHADOW_METRICS surface=multiscale setting=<L> psnr=<..> psnr_gain=<..> shadow_psnr=<..> \
    ssim=<..> mse=<..> full_psnr=<..>
```

- **psnr** (PRIMARY, higher better): SHADOW-REGION deshadow PSNR (dB), only over pixels the
  shadow touches. **shadow_psnr** is the copy floor; **psnr_gain** `= psnr − shadow_psnr` must
  be > 0. `full_psnr`, `ssim`, `mse` are diagnostics.

The pyramid gives higher aggregate shadow-region PSNR than the single-scale net (most on the
larger heavy shadows), and both clearly beat the shadowed-input identity floor.
