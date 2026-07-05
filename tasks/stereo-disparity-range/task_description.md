# Stereo Disparity Range: Sizing the Cost-Volume Search

## Research Question
A stereo cost volume enumerates candidate disparities `0..D_MAX-1`. The disparity
readout can only output a value within this range, so `D_MAX` must be large enough
to cover the largest disparity in the scene (the nearest objects). Given a FIXED
stereo net (features, cost volume, 3D aggregation, soft-argmin readout, loss,
schedule), **what maximum disparity minimizes EPE?**

## Background
If `D_MAX` is smaller than the true maximum disparity, the largest-disparity
(foreground) pixels cannot be represented; their prediction saturates at the top
of the range and their error is large — end-point error stays high. A range that
comfortably covers the scene's disparities lets the network represent every pixel.
Setting `D_MAX` too large wastes compute but does not hurt accuracy much. The
hardest setting's REAL ground-truth disparities span up to ~77 px.

## Implementation Contract
Modify `build_disp_range` in `stereo-matching/solution/drange.py`:

```python
def build_disp_range():
    # return the max disparity (levels), an int in [4, 128]
    ...
```

A malformed / crashing / out-of-bounds return degrades to the FIXED default
D_MAX for the active severity (see `SEVERITY_DMAX` in `harness.py`). The SAME
returned `D_MAX` is used across all 3 difficulty settings.

## Fixed Pipeline & Evaluation
- Data: REAL rectified stereo photographs from the Middlebury Stereo Datasets
  2005/2006 (Scharstein, Hirschmuller et al., CVPR 2007), with sub-pixel-accurate
  structured-light disparity ground truth (Scharstein & Szeliski, CVPR 2003) —
  fully open, no registration/login required. Fixed seed-42 128x256 crops (40
  train / 20 val) per setting; see
  `vendor/data_scripts/stereo-matching/prepare_data.py`.
- **3 difficulty settings** (the score is the geometric mean over all three) =
  27 real scenes ranked by ground-truth max disparity and split into terciles:
  - `easy`   — real scenes with ground-truth disparities up to ~59 px,
  - `medium` — real scenes with ground-truth disparities up to ~70 px,
  - `hard`   — real scenes with ground-truth disparities up to ~77 px.
- Model: a small GC-Net-style stereo net (features at 1/4 res, concatenation cost
  volume, 3D-conv aggregation, soft-argmin readout), trained with smooth-L1 +
  AdamW/OneCycle (1200 steps).
- Metric (LOWER is better): `epe_<setting>` — validation mean disparity
  end-point error in pixels.
- Scoring is anchored between a far-too-small range (8 px) and a range that
  comfortably covers every setting (96 px), per setting. Measured anchors are
  in `leaderboard.csv` (to be re-measured on GPU against the new real data;
  the current leaderboard.csv values are stale synthetic-data anchors).
