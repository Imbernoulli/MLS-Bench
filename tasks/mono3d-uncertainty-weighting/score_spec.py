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
The weak->strong order holds across all three settings and both seeds on real KITTI.

SOTA=0.5 anchor convention (matches stereo-disparity-range): "0.5 is the strongest baseline"
-- ref = the STRONG (weight_homoscedastic) baseline's SEED-42-SPECIFIC value (NOT
seed-averaged) so weight_homoscedastic scores exactly 0.5 at seed 42; scale =
(strong_seed42-weak_seed42)/ln(9) so the weak (weight_degenerate) baseline lands ~0.1 at
seed 42.
"""
from mlsbench.scoring.dsl import *

term("ap25_easy",
    col("ap25_easy").higher().id().sigmoid(ref=const(0.335036), scale=0.0298981728))
term("ap25_moderate",
    col("ap25_moderate").higher().id().sigmoid(ref=const(0.176036), scale=0.0144728037))
term("ap25_hard",
    col("ap25_hard").higher().id().sigmoid(ref=const(0.173538), scale=0.0109087620))

setting("easy", weighted_mean(("ap25_easy", 1.0)))
setting("moderate", weighted_mean(("ap25_moderate", 1.0)))
setting("hard", weighted_mean(("ap25_hard", 1.0)))

task(gmean("easy", "moderate", "hard"))
