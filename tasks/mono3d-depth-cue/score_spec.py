"""Score spec for mono3d-depth-cue.

Primary metric per setting: AP3D@0.25 = fraction of TEST objects (in this KITTI difficulty tier) whose
predicted 3D box has 3D IoU >= 0.25 with GT (HIGHER better; a degenerate predictor scores ~0).
The task score is the geometric mean of the per-setting AP3D over KITTI's own OFFICIAL
easy/moderate/hard difficulty tiers (bbox-height/occlusion/truncation thresholds), applied
EXCLUSIVELY as a disjoint partition of the held-out REAL KITTI test split.

The design lever is depth cue — box height vs width. MEASURED anchors (B0 8xH200, torch 2.4.1,
REAL KITTI 3D Object Detection, CROSS-SEED 42/123, 1200 steps, AP3D@0.25 per KITTI official
difficulty tier):
  cue_width (WEAK)         seed42 easy=0.170073 moderate=0.098807 hard=0.107383
  cue_height (STRONG)      seed42 easy=0.320438 moderate=0.164679 hard=0.171620
The yaw-confounded width cue robustly and by a wide margin underperforms the height cue on
every tier, both seeds (the mechanism transfers cleanly to real KITTI).

SOTA=0.5 anchor convention (matches stereo-disparity-range): "0.5 is the strongest baseline"
-- ref = the STRONG (cue_height) baseline's SEED-42-SPECIFIC value (NOT seed-averaged) so
cue_height scores exactly 0.5 at seed 42; scale = (strong_seed42-weak_seed42)/ln(9) so the
weak (cue_width) baseline lands ~0.1 at seed 42.
"""
from mlsbench.scoring.dsl import *

term("ap25_easy",
    col("ap25_easy").higher().id().sigmoid(ref=const(0.320438), scale=0.0684340606558722))
term("ap25_moderate",
    col("ap25_moderate").higher().id().sigmoid(ref=const(0.164679), scale=0.02997963916818151))
term("ap25_hard",
    col("ap25_hard").higher().id().sigmoid(ref=const(0.171620), scale=0.029235518600414068))

setting("easy", weighted_mean(("ap25_easy", 1.0)))
setting("moderate", weighted_mean(("ap25_moderate", 1.0)))
setting("hard", weighted_mean(("ap25_hard", 1.0)))

task(gmean("easy", "moderate", "hard"))
