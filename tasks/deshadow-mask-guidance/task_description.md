# Shadow Removal: Mask Guidance (Blind U-Net vs Mask-Guided Physical Deshadower)

## Objective

Design **the deshadower backbone** — a BLIND U-Net that sees only the shadowed RGB, or a
MASK-GUIDED U-Net that also takes the shadow mask as input — so that it removes a cast
shadow to match the clean, shadow-free ground truth as accurately as possible, maximizing
**shadow-region PSNR** on a held-out set of shadowed→clean pairs under a fixed formulation,
loss, optimizer, data, and evaluation protocol.

## Background

Shadow removal recovers a shadow-free image from one on which a **cast shadow**
multiplicatively darkens a known, soft-edged region. Under the physics-based linear
illumination model of Shadow Image Decomposition / SP+M-Net (Le et al., ICCV 2019) the
shadowed image is `I = a·J` with `a(x) = 1 − (1−att)·m(x)`, where `J` is the clean scene
(the GT), `m` is the soft shadow matte/mask (1 = umbra, 0 = lit, fractional = penumbra) and
`att` is the per-image umbra attenuation. Given the mask, the recovery is a spatially-varying
brightening — the net must know **where** the shadow is and **how much** to brighten:

- **unet_nomask** — a BLIND U-Net that sees only the 3-channel shadowed RGB and must both
  LOCATE and correct the shadow from colour alone (the DeshadowNet, Qu et al. CVPR 2017,
  multi-context intuition **without** the mask prior). It removes some shadow but, not
  knowing exactly where/how-much, leaks into the lit region and mis-corrects the soft
  penumbra — lower shadow-region PSNR.
- **unet_mask** — the MASK-GUIDED U-Net: the soft shadow mask is concatenated as a 4th input
  channel, so the net is told exactly WHERE and HOW MUCH to brighten — the SP+M-Net
  physically-parameterised recovery that fits the multiplicative attenuation. This is the
  strong answer and matches the literature ordering (mask-guided physical > blind deep net).

The blind net underfits the shadow boundary and scores lower shadow-region PSNR; the
mask-guided net exploits the provided mask and scores higher.

## Data & pipeline (FIXED)

- Tiny fixed set of REAL shadowed→shadow-free 64×64 photo triplets (+ a real shadow mask) from
  ISTD (Wang et al., CVPR 2018): the same outdoor scene photographed with and without a
  physical cast shadow. THREE severities (**light / medium / heavy**) are terciles of each
  triplet's MEASURED shadow attenuation (mean brightness ratio inside the mask). Train and val
  use ISTD's own **disjoint-scene train/test split**. The shadow-free photo is the ground
  truth. The shadow **mask is provided as an input**.
- A FIXED residual-learning formulation (`clean = shadowed + net(·)`), a fixed composite
  loss (L1 + SSIM, up-weighted inside the shadow), a fixed optimizer, trained a few hundred
  steps, fixed seed. Only your backbone / mask-usage varies.

## Implementation Contract

Edit **only** `get_network_config()` in `image-deshadow/solution/network.py`. Return a dict:

```python
def get_network_config():
    # {'arch': 'copy' | 'unet_nomask' | 'unet_mask'}
    return {"arch": "unet_mask"}
```

A malformed / crashing return falls back to `arch="unet_mask"`.

## Metric

The harness prints one line per run:

```
DESHADOW_METRICS surface=network setting=<L> psnr=<..> psnr_gain=<..> shadow_psnr=<..> \
    ssim=<..> mse=<..> full_psnr=<..>
```

- **psnr** (PRIMARY, higher better): SHADOW-REGION deshadow PSNR (dB) of the **deshadowed
  output vs the clean GT**, computed **only over pixels the shadow touches** — so a method
  that merely copies the LIT region gains nothing; it must actually brighten the shadow.
- **shadow_psnr**: PSNR of the **shadowed input** over the shadow region vs the clean GT —
  the copy / do-nothing floor. A net that copies its input scores `psnr == shadow_psnr`.
- **psnr_gain** `= psnr − shadow_psnr`: the deshadowed output must **beat passing the
  shadowed input through** (gain > 0). A constant / all-white / all-black output scores far
  **below** this floor.
- `full_psnr` (whole image), `ssim`, `mse` are diagnostics only.

The blind U-Net removes some shadow (positive gain) but the mask-guided U-Net gives clear
further shadow-region PSNR headroom, while both beat the shadowed-input identity floor.
