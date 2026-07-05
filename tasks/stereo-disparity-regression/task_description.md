# Stereo Disparity Regression: Soft-Argmin vs Argmax Readout

## Research Question
A stereo-matching network builds a **disparity cost volume** — for each pixel a
cost curve over `D` candidate disparities — and must turn that curve into a single
per-pixel disparity. GC-Net's key contribution is the differentiable
**soft-argmin**: apply softmax over the `D` costs to get a probability
distribution, then output its **expectation** `sum_d d * p(d)`. Given a FIXED
stereo net (features, cost volume, 3D aggregation, loss, schedule), **what
disparity readout minimizes EPE?**

## Background
A hard **argmax** (winner-take-all) picks the single lowest-cost integer disparity
level. It is not differentiable, has no sub-pixel accuracy, and passes no gradient
back to the cost volume, so the network cannot learn to shape its cost curves —
end-point error stays high. The soft-argmin expectation is differentiable and
sub-pixel accurate, so training works and EPE drops substantially.

## Implementation Contract
Modify `build_regressor` in `stereo-matching/solution/regress.py`:

```python
def build_regressor():
    # return regress(cost, disp_values) -> (B,H,W) disparity
    ...
```

`cost` is the aggregated cost volume `(B, D, H, W)` (LARGER = more likely match at
that level); `disp_values` is a `(D,)` tensor of the disparity (px) of each level.
A malformed / crashing return degrades to the FIXED soft-argmin reference.

## Fixed Pipeline & Evaluation
- Data: REAL rectified stereo photographs from the Middlebury Stereo Datasets
  2005/2006 (Scharstein, Hirschmuller et al., CVPR 2007), with sub-pixel-accurate
  structured-light disparity ground truth (Scharstein & Szeliski, CVPR 2003) —
  fully open, no registration/login required. Fixed seed-42 128x256 crops (40
  train / 20 val) per setting; see
  `vendor/data_scripts/stereo-matching/prepare_data.py`.
- **3 difficulty settings** (the score is the geometric mean over all three)
  = 27 real scenes ranked by ground-truth max disparity and split into
  terciles:
  - `easy`   — real scenes with ground-truth disparities up to ~59 px,
  - `medium` — real scenes with ground-truth disparities up to ~70 px,
  - `hard`   — real scenes with ground-truth disparities up to ~77 px.
- Model: a small GC-Net-style stereo net (shared 2D feature extractor at 1/4 res,
  concatenation cost volume, 3D-conv aggregation), trained with smooth-L1 and
  AdamW + OneCycle for a short fixed schedule (1200 steps).
- Metric (LOWER is better): `epe_<setting>` — validation mean disparity
  end-point error in pixels.
- Scoring is anchored between the hard-argmax readout and the soft-argmin
  expectation, per setting: you score well by using a differentiable, sub-pixel
  disparity readout. Measured anchors are in `leaderboard.csv`.
