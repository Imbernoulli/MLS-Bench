# Image Harmonization: Region-Conditioning the Harmonizer (Mask-Blind vs Mask-Conditioned)

## Objective

Design **the harmonizer network** — a mask-blind global image-to-image U-Net, or a
mask-conditioned U-Net that is told which region is the pasted foreground — so that it
**recolours a composited foreground region to match the background** (photometric
consistency) as accurately as possible, maximizing the **foreground-region PSNR** of the
harmonized output vs the real ground truth on a held-out set of REAL iHarmony4 composites,
under a fixed data pipeline, loss, optimizer, and evaluation protocol.

## Background

Image harmonization adjusts the **appearance** (colour / brightness / white balance) of a
pasted foreground so it looks like it belongs in the scene. Unlike inpainting (fill a
hole), matting (extract an alpha), colorization (grey → colour) or dehazing (remove a
global scattering degradation), here the foreground content is **already present and
structurally correct** — only its **colour statistics** are wrong (a region pasted from a
different photo). The core inductive bias of harmonization is that the recolour must be
applied to the **pasted foreground** while the (already-correct) **background is
preserved** (Cong et al., DoveNet, CVPR 2020, which introduced the iHarmony4 benchmark;
Ling et al., RainNet, CVPR 2021):

- **blind** — a MASK-BLIND encoder-decoder U-Net (composite RGB only, 3-channel input). It
  is **region-agnostic**: it cannot tell the foreground from the background, so it applies
  a compromise correction that disturbs the already-correct background and only partially
  fixes the foreground → a middling foreground PSNR.
- **mask** — the MASK-CONDITIONED U-Net (composite RGB **+ the foreground mask**, 4-channel
  input). It knows exactly which region is the pasted foreground and recolours only it
  while preserving the background — the mask-conditioning every real harmonizer relies on →
  the highest foreground PSNR.

The mask-blind net under-corrects and scores lower; the mask-conditioned net targets the
foreground and scores higher. Both clearly beat the input-copy (do-nothing) floor.

## Data & pipeline (FIXED)

- REAL composites from the **iHarmony4** benchmark (Cong et al., DoveNet, CVPR
  2020): a (composite, foreground mask, real photo) triple per example, resized to 64×64.
  The composite's foreground was colour-transferred by the iHarmony4 authors against a
  real reference photo (NOT a synthetic knob); the un-shifted real photo is the EXACT
  harmonized ground truth, and the metric is measured **only inside the foreground
  region** (the background is already correct). Each sub-dataset's OFFICIAL disjoint
  train/test id split is used (no leakage).
- Three **settings** = three of iHarmony4's REAL sub-datasets, used as `mild` /
  `medium` / `strong` by MEASURED foreground composite-vs-GT PSNR floor at this
  harness's fixed 64×64 working resolution: `mild`=HCOCO (2000 train/400 val),
  `medium`=Hday2night (200 train/80 val — capped by the sub-dataset's small size),
  `strong`=HFlickr (2000 train/400 val); the fixed harness, base width/depth, L1 loss, optimizer, iterations and seed
  are the same across all three. Only your architecture varies.

## Implementation Contract

Edit **only** `get_network_config()` in `image-harmonization/solution/network.py`. Return a
dict:

```python
def get_network_config():
    # {'arch': 'copy' | 'blind' | 'mask' | 'rain'}
    return {"arch": "mask"}
```

The default returns `{"arch": "blind"}` (mask-blind — partial recovery, lower PSNR). A
malformed / crashing return falls back to `arch="mask"`.

## Metric

The harness prints one line per run:

```
HARMONY_METRICS surface=network setting=<L> fg_psnr=<..> fg_psnr_gain=<..> \
    comp_fg_psnr=<..> fg_mse=<..> fg_ssim=<..>
```

- **fg_psnr** (PRIMARY, higher better): foreground-region PSNR (dB) of the **harmonized
  output vs the real GT**, measured ONLY inside the foreground mask.
- **comp_fg_psnr**: foreground-region PSNR of the **composite input** vs the GT — the
  identity ("do-nothing") floor. The `copy` identity scores `fg_psnr == comp_fg_psnr`.
- **fg_psnr_gain** `= fg_psnr − comp_fg_psnr`: the output must **beat copying the composite
  through** (gain > 0).
- `fg_mse` / `fg_ssim` are diagnostics only.

The mask-blind net under-corrects (lower `fg_psnr`); the mask-conditioned net gives clear
PSNR headroom above it while all clearly beat the do-nothing floor. Scoring is anchored
per severity between the blind (weak) and mask (strong) baselines; measured anchors are
recorded in `leaderboard.csv`.
