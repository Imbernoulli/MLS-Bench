# Shadow Removal: Physics Parameterisation (Free Residual vs SP+M-Net Illumination Model)

## Objective

Design **how the deshadower produces its output** — predict a free, unconstrained 3-channel
residual, or predict the **SP+M-Net physical illumination parameters** and apply the
multiplicative-illumination inverse — so that the mask-guided deshadower removes a cast shadow
to match the clean, shadow-free ground truth as accurately as possible, maximizing
**shadow-region PSNR** on a held-out set of shadowed→clean pairs under a fixed backbone, loss,
optimizer, data, and evaluation protocol.

## Background

Under the physics-based linear illumination model of Shadow Image Decomposition / SP+M-Net
(Le et al., ICCV 2019) the shadowed image is `I = a·J` with `a(x) = 1 − (1−att)·m(x)`. The
clean image is therefore a **per-pixel affine relighting** of the shadowed one, `J = w·I + b`.
A net that predicts a free residual is unconstrained — it can drift the already-lit region and
does not respect this multiplicative structure. A net that predicts the affine `(w, b)` and
outputs `w·I + b` produces a valid illumination inverse **by construction**.

- **weak (`mode='residual'`)** — predict a free 3-channel RESIDUAL added to the shadowed input
  (`clean = shadowed + net(·)`). Unconstrained; does not respect the multiplicative illumination
  structure.
- **strong (`mode='physics'`)** — the SP+M-Net ILLUMINATION MODEL: the net predicts per-pixel
  affine relighting parameters `(w, b)` and outputs `J = w·I + b`, a valid
  multiplicative-illumination inverse initialised near identity, matching the true degradation
  form.

## Honesty note

On this synthetic shadow the physics parameterisation's benefit is **concentrated in the easier
settings**: a large gain on **light** and a solid gain on **medium**, but on the **heavy**
setting it is a near-tie (on one seed the affine output marginally under-shoots the deepest
umbra). The TASK-LEVEL geometric mean of shadow-region PSNR is monotone strong > weak on **both**
validated seeds, so the surface is a valid weak→strong lever at the score level; the light/medium
gains dominate the aggregate.

## Data & pipeline (FIXED)

- Tiny fixed set of shadowed→clean 64×64 pairs (+ a soft shadow mask) at THREE cast-shadow
  severities (**light / medium / heavy**). Train and val use **disjoint patches AND disjoint
  shadow realisations**. The clean patch is the **exact** ground truth. The soft shadow **mask
  is provided as an input**.
- A FIXED mask-guided deshadower backbone, a fixed composite L1+SSIM loss up-weighted inside
  the shadow, a fixed optimizer, trained a few hundred steps, fixed seed. Only the output
  parameterisation varies.

## Implementation Contract

Edit **only** `get_physics_config()` in `image-deshadow/solution/physics.py`. Return a dict:

```python
def get_physics_config():
    # {'mode': 'residual' | 'physics'}
    return {"mode": "physics"}
```

A malformed / crashing return falls back to `mode='residual'` (weak).

## Metric

```
DESHADOW_METRICS surface=physics setting=<L> psnr=<..> psnr_gain=<..> shadow_psnr=<..> \
    ssim=<..> mse=<..> full_psnr=<..>
```

- **psnr** (PRIMARY, higher better): SHADOW-REGION deshadow PSNR (dB), only over pixels the
  shadow touches. **shadow_psnr** is the copy floor; **psnr_gain** `= psnr − shadow_psnr` must
  be > 0. `full_psnr`, `ssim`, `mse` are diagnostics.

The SP+M-Net physics parameterisation gives higher aggregate shadow-region PSNR than the free
residual (most on the lighter shadows), and both clearly beat the shadowed-input identity floor.
