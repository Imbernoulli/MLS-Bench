# Stereo Loss Design: Smooth-L1 vs Squared-L2 Disparity Regression

## Research Question
A stereo network regresses a per-pixel disparity from its cost volume and is
trained with a regression loss on that disparity. The per-pixel disparity error is
dominated by a few large-error pixels (occlusions, depth discontinuities). Given a
FIXED stereo net (features, cost volume, 3D aggregation, soft-argmin readout,
schedule), **what regression loss minimizes EPE?**

## Background
A **squared-L2** loss squares the large errors, so their gradient dominates,
destabilises training and biases toward an over-smoothed mean disparity —
end-point error stays high. GC-Net and PSMNet deliberately use the robust
**smooth-L1** (Huber) loss, which is quadratic for small (sub-pixel) errors but
linear for large ones, so it keeps sub-pixel precision while being outlier-robust —
EPE drops.

## Implementation Contract
Modify `build_loss` in `stereo-matching/solution/loss.py`:

```python
def build_loss():
    # return loss_fn(disp_pred, disp_gt, valid) -> scalar tensor
    ...
```

`disp_pred`, `disp_gt` are `(B, H, W)` disparity maps; `valid` is a `(B, H, W)`
mask (1 = valid). A malformed / crashing / non-finite return degrades to the FIXED
smooth-L1 reference.

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
- Model: a small GC-Net-style stereo net (features at 1/4 res, concatenation cost
  volume, 3D-conv aggregation, soft-argmin readout), trained with AdamW/OneCycle
  (1200 steps).
- Metric (LOWER is better): `epe_<setting>` — validation mean disparity
  end-point error in pixels.
- Scoring is anchored between the squared-L2 loss and the smooth-L1 loss, per
  setting: you score well by using a robust regression loss. Measured anchors
  are in `leaderboard.csv`.
