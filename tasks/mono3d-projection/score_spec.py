"""Score spec for mono3d-projection.

Primary metric per setting: AP3D@0.25 = fraction of TEST objects (in this KITTI difficulty tier) whose
predicted 3D box has 3D IoU >= 0.25 with GT (HIGHER better; a degenerate predictor scores ~0).
The task score is the geometric mean of the per-setting AP3D over KITTI's own OFFICIAL
easy/moderate/hard difficulty tiers (bbox-height/occlusion/truncation thresholds), applied
EXCLUSIVELY as a disjoint partition of the held-out REAL KITTI test split.

The design lever is 3d center back-projection. MEASURED anchors (B0 8xH200, torch 2.4.1,
REAL KITTI 3D Object Detection, CROSS-SEED 42/123, seed-averaged, 1200 steps, AP3D@0.25 per
KITTI official difficulty tier):
  proj_onaxis (WEAK)       easy=0.0000  moderate=0.0000  hard=0.0000
  proj_pinhole (STRONG)    easy=0.3420  moderate=0.1650  hard=0.1783
The degenerate on-axis (no-perspective) back-projection scores EXACTLY 0 AP3D on every tier,
both seeds -- an unambiguous, maximally robust floor. Per-setting logistic midpoint = the
strong (proj_pinhole) reference -> score 0.5, scale = (strong-weak)/ln(9) so the weak
(proj_onaxis) baseline lands ~0.1.
"""
from mlsbench.scoring.dsl import *

term("ap25_easy",
    col("ap25_easy").higher().id().sigmoid(ref=const(0.341971), scale=0.1556377092844031))
term("ap25_moderate",
    col("ap25_moderate").higher().id().sigmoid(ref=const(0.164963), scale=0.07507789677102149))
term("ap25_hard",
    col("ap25_hard").higher().id().sigmoid(ref=const(0.178332), scale=0.08116216332160192))

setting("easy", weighted_mean(("ap25_easy", 1.0)))
setting("moderate", weighted_mean(("ap25_moderate", 1.0)))
setting("hard", weighted_mean(("ap25_hard", 1.0)))

task(gmean("easy", "moderate", "hard"))
