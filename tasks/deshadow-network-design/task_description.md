# Shadow Removal: Deshadower Design (Do-Nothing Copy vs Mask-Guided Deshadower)

## Objective

Design **the deshadower** — either pass the shadowed image through unchanged (no removal),
or a MASK-GUIDED U-Net that removes the cast shadow — so that the output matches the clean,
shadow-free ground truth as accurately as possible, maximizing **shadow-region PSNR** on a
held-out set of shadowed→clean pairs under a fixed formulation, loss, optimizer, data, and
evaluation protocol.

## Background

Shadow removal recovers a shadow-free image from one on which a **cast shadow**
multiplicatively darkens a known, soft-edged region. Under the physics-based linear
illumination model of Shadow Image Decomposition / SP+M-Net (Le et al., ICCV 2019) the
shadowed image is `I = a·J` with `a(x) = 1 − (1−att)·m(x)` (`J` = clean GT, `m` = soft shadow
matte/mask, `att` = umbra attenuation). The soft shadow **mask is provided as an input**.

- **copy** — pass the shadowed input straight through. This does NOTHING to the shadow: it
  scores exactly the shadowed-input shadow-region PSNR (the do-nothing floor). Any real
  deshadower must beat it.
- **unet_mask** — the MASK-GUIDED U-Net (residual learning `clean = shadowed + net(·)`): the
  soft shadow mask is concatenated as a 4th input channel, so the net knows exactly WHERE
  and HOW MUCH to brighten — the SP+M-Net physically-parameterised recovery that fits the
  multiplicative attenuation. It gives a large shadow-region PSNR gain over the do-nothing
  floor.

## Data & pipeline (FIXED)

- Tiny fixed set of REAL shadowed→shadow-free 64×64 photo triplets (+ a real shadow mask) from
  ISTD (Wang et al., CVPR 2018): the same outdoor scene photographed with and without a
  physical cast shadow. THREE severities (**light / medium / heavy**) are terciles of each
  triplet's MEASURED shadow attenuation (mean brightness ratio inside the mask). Train and val
  use ISTD's own **disjoint-scene train/test split**. The shadow-free photo is the ground
  truth.
- A FIXED residual-learning formulation, a fixed composite loss (L1 + SSIM, up-weighted
  inside the shadow), a fixed optimizer, trained a few hundred steps, fixed seed. Only your
  backbone choice varies.

## Implementation Contract

Edit **only** `get_network_config()` in `image-deshadow/solution/network.py`. Return a dict:

```python
def get_network_config():
    # {'arch': 'copy' | 'unet_nomask' | 'unet_mask'}
    return {"arch": "unet_mask"}
```

The default returns `arch="copy"` (the do-nothing floor). A malformed / crashing return
falls back to `arch="unet_mask"`.

## Metric

The harness prints one line per run:

```
DESHADOW_METRICS surface=network setting=<L> psnr=<..> psnr_gain=<..> shadow_psnr=<..> \
    ssim=<..> mse=<..> full_psnr=<..>
```

- **psnr** (PRIMARY, higher better): SHADOW-REGION deshadow PSNR (dB) of the **output vs the
  clean GT**, computed **only over pixels the shadow touches** — so a method that merely
  copies the LIT region gains nothing; it must actually brighten the shadow.
- **shadow_psnr**: PSNR of the **shadowed input** over the shadow region — the copy /
  do-nothing floor. The `copy` backbone scores `psnr == shadow_psnr` (gain 0).
- **psnr_gain** `= psnr − shadow_psnr`: the output must **beat passing the shadowed input
  through** (gain > 0). A constant / all-white / all-black output scores far **below** this
  floor.
- `full_psnr`, `ssim`, `mse` are diagnostics only.

The mask-guided U-Net gives a large shadow-region PSNR headroom above the do-nothing floor.
