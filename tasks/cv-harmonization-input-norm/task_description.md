# Image Harmonization: Background-Referenced Input Normalization (Whiten vs Raw)

## Objective

Design **whether to apply a fixed input normalization** to the harmonizer — a whitening of
the whole composite by the BACKGROUND per-channel mean/std, or feeding the raw composite —
so that the harmonizer recolours the composited foreground region to match the background
as accurately as possible. Maximize the **foreground-region PSNR** of the harmonized output
vs the real ground truth on a held-out set of REAL iHarmony4 composites, under a fixed data
pipeline, loss, optimizer, and evaluation protocol. Only **the fixed background-referenced
input normalization** varies; every other design axis is fixed at the strong reference
config.

## Background

Image harmonization adjusts the **appearance** (colour / brightness / white balance) of a
pasted foreground so it looks like it belongs in the scene. Unlike inpainting (fill a
hole), matting (extract an alpha), colorization (grey → colour) or dehazing (remove a
global scattering degradation), the foreground content is **already present and
structurally correct** — only its **colour statistics** are wrong (Cong et al., DoveNet,
CVPR 2020; Ling et al., RainNet, CVPR 2021).

A tempting idea is to **whiten** the whole composite by the BACKGROUND per-channel mean/std
before the network (and un-whiten the output), so the foreground/background appearance gap
is expressed relative to the already-correct background. In practice this **fixed**
whitening is a poor input transform here: it rescales the image by per-image background
statistics that vary wildly, destroying the absolute colour levels the harmonizer needs and
injecting instability at the un-whitening step — the reconstruction collapses. Feeding the
**raw composite** is far more robust. This task varies that choice:

- **`bg_whiten`** — whiten the whole image by the BACKGROUND per-channel mean/std, then
  un-whiten the output. The naive background-referencing transform: it corrupts the input
  colour scale and the reconstruction collapses (much lower foreground PSNR).
- **`none`** — feed the **raw composite** (the net handles the colour levels directly). The
  robust choice → much higher foreground PSNR.

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

Edit **only** `get_input_norm()` in `image-harmonization/solution/inputnorm.py`. Return
`{"none" | "bg_whiten"}`:

```python
def get_input_norm():
    # {"none" | "bg_whiten"}
    return "none"
```

The default returns `"bg_whiten"` (the naive, degrading choice). A malformed / crashing
return falls back to the robust reference (`"none"`).

## Metric

The harness prints one line per run:

```
HARMONY_METRICS surface=inputnorm setting=<L> fg_psnr=<..> fg_psnr_gain=<..> \
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
settings), anchored per severity between the weak (`bg_whiten`) and strong (`none`)
baselines; measured anchors are recorded in `leaderboard.csv`.
