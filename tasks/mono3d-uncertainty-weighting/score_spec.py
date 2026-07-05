"""Score spec for mono3d-uncertainty-weighting.

Primary metric per setting: AP3D@0.25 = fraction of TEST objects (in this KITTI difficulty tier) whose
predicted 3D box has 3D IoU >= 0.25 with GT (HIGHER better; a degenerate predictor scores ~0).
The task score is the geometric mean of the per-setting AP3D over KITTI's own OFFICIAL
easy/moderate/hard difficulty tiers (bbox-height/occlusion/truncation thresholds), applied
EXCLUSIVELY as a disjoint partition of the held-out REAL KITTI test split.

The design lever is multi-task uncertainty weighting. MEASURED anchors (B0 8xH200, torch 2.4.1,
REAL KITTI 3D Object Detection, CROSS-SEED 42/123, seed-averaged, 1200 steps, AP3D@0.25 per
KITTI official difficulty tier):
  weight_degenerate (WEAK)      easy=0.2613  moderate=0.1408  hard=0.1405
  weight_homoscedastic (STRONG) easy=0.3336  moderate=0.1735  hard=0.1663
Per-setting logistic midpoint = the strong (weight_homoscedastic) reference -> score 0.5, scale
= (strong-weak)/ln(9) so the weak (weight_degenerate) baseline lands ~0.1; the weak->strong
order holds across all three settings and both seeds on real KITTI.
"""
from mlsbench.scoring.dsl import *

term("ap25_easy",
    col("ap25_easy").higher().id().sigmoid(ref=const(0.333576), scale=0.03288808105706092))
term("ap25_moderate",
    col("ap25_moderate").higher().id().sigmoid(ref=const(0.173481), scale=0.014860565613909752))
term("ap25_hard",
    col("ap25_hard").higher().id().sigmoid(ref=const(0.166347), scale=0.011781453870037815))

setting("easy", weighted_mean(("ap25_easy", 1.0)))
setting("moderate", weighted_mean(("ap25_moderate", 1.0)))
setting("hard", weighted_mean(("ap25_hard", 1.0)))

task(gmean("easy", "moderate", "hard"))
