# Shadow Removal: Mask Guidance (Blind vs Mask-Guided Deshadower)

## Objective

Design **whether the soft shadow mask is fed to the deshadower** — a BLIND U-Net that sees
only the shadowed RGB, or a MASK-GUIDED U-Net that concatenates the shadow mask as a 4th
input channel — so that it removes a cast shadow to match the clean, shadow-free ground truth
as accurately as possible, maximizing **shadow-region PSNR** on a held-out set of
shadowed→clean pairs under a fixed formulation, loss, optimizer, data, and evaluation
protocol.

## Background

Shadow removal recovers a shadow-free image from one on which a **cast shadow**
multiplicatively darkens a known, soft-edged region. Under the SP+M-Net physics (Le et al.,
ICCV 2019) the shadowed image is `I = a·J`, `a(x) = 1 − (1−att)·m(x)`. Given the mask, the
recovery is a spatially-varying brightening — the net must know **where** the shadow is and
**how much** to brighten:

- **weak (`use_mask=False`)** — a BLIND U-Net that sees only the 3-channel shadowed RGB and
  must both LOCATE and correct the shadow from colour alone (DeshadowNet, Qu et al. CVPR 2017,
  **without** the mask prior). It leaks into the lit region and mis-corrects the soft penumbra
  — lower shadow-region PSNR.
- **strong (`use_mask=True`)** — the MASK-GUIDED U-Net: the soft shadow mask is concatenated
  as a 4th input channel, so the net is told exactly WHERE and HOW MUCH to brighten — the
  SP+M-Net physically-parameterised recovery. Higher shadow-region PSNR.

This is a focused re-framing of `deshadow-network-design` / `deshadow-mask-guidance` on the
same validated harness: the ONLY lever here is the mask-input toggle; the residual formulation,
composite loss, backbone, optimizer, data, iterations, and seed are all FIXED.

## Data & pipeline (FIXED)

- Tiny fixed set of REAL shadowed→shadow-free 64×64 photo triplets (+ a real shadow mask) from
  ISTD (Wang et al., CVPR 2018): the same outdoor scene photographed with and without a
  physical cast shadow. THREE severities (**light / medium / heavy**) are terciles of each
  triplet's MEASURED shadow attenuation (mean brightness ratio inside the mask). Train and val
  use ISTD's own **disjoint-scene train/test split**. The shadow-free photo is the ground
  truth. The shadow **mask is provided as an input** (whether the net USES it is the lever).
- A FIXED residual-learning deshadower (`clean = shadowed + net(·)`), a fixed composite loss
  (L1 + SSIM, up-weighted inside the shadow), a fixed optimizer, trained a few hundred steps,
  fixed seed. Only the mask-input toggle varies.

## Implementation Contract

Edit **only** `get_mask_config()` in `image-deshadow/solution/mask.py`. Return a dict:

```python
def get_mask_config():
    # {'use_mask': True | False}
    return {"use_mask": True}
```

A malformed / crashing return falls back to `use_mask=False` (weak).

## Metric

The harness prints one line per run:

```
DESHADOW_METRICS surface=mask setting=<L> psnr=<..> psnr_gain=<..> shadow_psnr=<..> \
    ssim=<..> mse=<..> full_psnr=<..>
```

- **psnr** (PRIMARY, higher better): SHADOW-REGION deshadow PSNR (dB), computed **only over
  pixels the shadow touches**.
- **shadow_psnr**: the shadowed-input copy floor. **psnr_gain** `= psnr − shadow_psnr` must
  be > 0.
- `full_psnr`, `ssim`, `mse` are diagnostics only.

The mask-guided net gives higher shadow-region PSNR than the blind net, and both clearly beat
the shadowed-input identity floor.
