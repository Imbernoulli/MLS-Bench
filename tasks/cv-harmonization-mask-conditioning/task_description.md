# Image Harmonization: How the Foreground Mask is Fed (Region Conditioning)

## Objective

Design **how the harmonizer is told which region is the pasted foreground** — mask-blind, mask-concatenated, or mask-concatenated with a mask-gated output blend — so that it **recolours a composited foreground region to match the background** as accurately as possible. Maximize the **foreground-region PSNR** of the harmonized output vs the
real ground truth on a held-out set of REAL iHarmony4 composites, under a fixed data pipeline,
loss, optimizer, and evaluation protocol. Only **how the foreground **mask is conditioned into the network** (region conditioning)** varies; every other design
axis is fixed at the strong reference config.

## Background

Image harmonization adjusts the **appearance** (colour / brightness / white balance) of a
pasted foreground so it looks like it belongs in the scene. Unlike inpainting (fill a
hole), matting (extract an alpha), colorization (grey → colour) or dehazing (remove a
global scattering degradation), the foreground content is **already present and
structurally correct** — only its **colour statistics** are wrong. The core inductive
bias is that the recolour must be applied to the **pasted foreground** while the
(already-correct) **background is preserved** (Cong et al., DoveNet, CVPR 2020 (iHarmony4 benchmark); Ling et al., RainNet — Region-aware Adaptive Instance Normalization for Image Harmonization, CVPR 2021). This task varies how the foreground **mask is conditioned into the network** (region conditioning):

- **`none`** — **mask-BLIND** — the net sees only the composite RGB. Region-agnostic; it cannot tell the foreground from the background, applies a compromise correction → middling PSNR.
- **`concat`** — the foreground mask is **concatenated as a 4th input channel** (DoveNet, Cong et al. CVPR 2020) → the net recolours the correct region.
- **`gated`** — `concat` **plus a mask-gated output blend** that hard-restricts the edit to the foreground → the background is provably preserved and all capacity goes to the foreground (strongest).

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
  `strong`=HFlickr (2000 train/400 val); the fixed harness, base width/depth, optimizer, iterations and seed are the
  same across all three. Only your chosen surface varies; every other design axis is fixed
  at the strong reference config.

## Implementation Contract

Edit **only** `get_mask_conditioning()` in `image-harmonization/solution/maskcond.py`. Return
`{"none" | "concat" | "gated"}`:

```python
def get_mask_conditioning():
    # {"none" | "concat" | "gated"}
    return "gated"
```

The default returns `"none"` (the weak choice). A malformed / crashing return
falls back to the strong reference.

## Metric

The harness prints one line per run:

```
HARMONY_METRICS surface=maskcond setting=<L> fg_psnr=<..> fg_psnr_gain=<..> \
    comp_fg_psnr=<..> fg_mse=<..> fg_ssim=<..>
```

- **fg_psnr** (PRIMARY, higher better): foreground-region PSNR (dB) of the **harmonized
  output vs the real GT**, measured ONLY inside the foreground mask.
- **comp_fg_psnr**: foreground-region PSNR of the **composite input** vs the GT — the
  identity ("do-nothing") floor. A copy-composite output scores `fg_psnr == comp_fg_psnr`.
- **fg_psnr_gain** `= fg_psnr − comp_fg_psnr`: the output must **beat copying the composite
  through** (gain > 0).
- `fg_mse` / `fg_ssim` are diagnostics only.

Scoring aggregates the per-severity foreground PSNR (geometric mean over the three
settings), anchored per severity between the weak and strong baselines; measured anchors
are recorded in `leaderboard.csv`.
