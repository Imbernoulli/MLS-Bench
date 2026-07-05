"""Score spec for mono3d-depth-cue.

Primary metric per setting: AP3D@0.25 = fraction of TEST objects (in this KITTI difficulty tier) whose
predicted 3D box has 3D IoU >= 0.25 with GT (HIGHER better; a degenerate predictor scores ~0).
The task score is the geometric mean of the per-setting AP3D over KITTI's own OFFICIAL
easy/moderate/hard difficulty tiers (bbox-height/occlusion/truncation thresholds), applied
EXCLUSIVELY as a disjoint partition of the held-out REAL KITTI test split.

The design lever is depth cue — box height vs width. MEASURED anchors (B0 8xH200, torch 2.4.1,
REAL KITTI 3D Object Detection, CROSS-SEED 42/123, seed-averaged, 1200 steps, AP3D@0.25 per
KITTI official difficulty tier):
  cue_width (WEAK)         easy=0.1734  moderate=0.0957  hard=0.0997
  cue_height (STRONG)      easy=0.3259  moderate=0.1661  hard=0.1702
The yaw-confounded width cue robustly and by a wide margin underperforms the height cue on
every tier, both seeds (the mechanism transfers cleanly to real KITTI). Per-setting logistic
midpoint = the strong (cue_height) reference -> score 0.5, scale = (strong-weak)/ln(9) so the
weak (cue_width) baseline lands ~0.1.
"""
from mlsbench.scoring.dsl import *

term("ap25_easy",
    col("ap25_easy").higher().id().sigmoid(ref=const(0.325912), scale=0.06943077260902858))
term("ap25_moderate",
    col("ap25_moderate").higher().id().sigmoid(ref=const(0.166098), scale=0.032047020011657705))
term("ap25_hard",
    col("ap25_hard").higher().id().sigmoid(ref=const(0.170182), scale=0.03207205159038995))

setting("easy", weighted_mean(("ap25_easy", 1.0)))
setting("moderate", weighted_mean(("ap25_moderate", 1.0)))
setting("hard", weighted_mean(("ap25_hard", 1.0)))

task(gmean("easy", "moderate", "hard"))
