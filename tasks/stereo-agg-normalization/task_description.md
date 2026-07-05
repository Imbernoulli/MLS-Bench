# Stereo Cost-Aggregation Normalization: None vs BatchNorm3d

## Research Question
This task studies ONE stereo-matching design decision on a FIXED small
GC-Net/PSMNet-style network (features, cost volume, 3D aggregation, soft-argmin
readout, loss, schedule all fixed apart from this surface): what maximizes
disparity accuracy (minimizes EPE)? Reference: Kendall et al. ICCV 2017 (GC-Net).

## Implementation Contract
Edit the marked region of `stereo-matching/solution/normalization.py` (surface `normalization`).
The weak default is the `none` baseline; the strong/SOTA reference is the
`batch` baseline.

## Fixed Pipeline & Evaluation
- Data: REAL rectified stereo photographs from the Middlebury Stereo
  Datasets 2005/2006 (Scharstein, Hirschmuller et al., CVPR 2007), with
  sub-pixel-accurate structured-light disparity ground truth (Scharstein &
  Szeliski, CVPR 2003) — fully open, no registration/login required. Fixed
  seed-42 128x256 crops (40 train / 20 val) per setting; see
  `vendor/data_scripts/stereo-matching/prepare_data.py`.
- **3 difficulty settings** (the score is the geometric mean over all three)
  = 27 real scenes ranked by ground-truth max disparity and split into
  terciles:
  - `easy`   — real scenes with ground-truth disparities up to ~59 px,
  - `medium` — real scenes with ground-truth disparities up to ~70 px,
  - `hard`   — real scenes with ground-truth disparities up to ~77 px.
- Model: a small GC-Net/PSMNet-style stereo net, trained with AdamW + OneCycle
  for a short fixed schedule (1200 steps).
- Metric (LOWER is better): `epe_<setting>` — validation mean disparity
  end-point error in pixels.
- Scoring is anchored between the weak (`none`) and strong (`batch`)
  baselines per setting. Measured anchors are in `leaderboard.csv`.
