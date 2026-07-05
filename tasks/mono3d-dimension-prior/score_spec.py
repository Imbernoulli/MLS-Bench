"""Score spec for mono3d-dimension-prior.

Primary metric per setting: AP3D@0.25 = fraction of TEST objects (in this KITTI difficulty tier) whose
predicted 3D box has 3D IoU >= 0.25 with GT (HIGHER better; a degenerate predictor scores ~0).
The task score is the geometric mean of the per-setting AP3D over KITTI's own OFFICIAL
easy/moderate/hard difficulty tiers (bbox-height/occlusion/truncation thresholds), applied
EXCLUSIVELY as a disjoint partition of the held-out REAL KITTI test split.

The design lever is dimension prior / shape anchors. MEASURED anchors (B0 8xH200, torch 2.4.1,
REAL KITTI 3D Object Detection, CROSS-SEED 42/123, seed-averaged, 1200 steps, AP3D@0.25 per
KITTI official difficulty tier):
  dims_direct (WEAK)       easy=0.3139  moderate=0.1573  hard=0.1438
  dims_prior (STRONG)      easy=0.3347  moderate=0.1780  hard=0.1683
The class-conditional dimension prior robustly beats direct dimension regression on every
tier, both seeds (modest but consistent margins, 0.009-0.025 AP3D on real KITTI).

SOTA=0.5 anchor convention (matches stereo-disparity-range): "0.5 is the strongest baseline"
-- ref = the STRONG (dims_prior) baseline's SEED-42-SPECIFIC value (NOT seed-averaged) so
dims_prior scores exactly 0.5 at seed 42; scale = (strong_seed42-weak_seed42)/ln(9) so the
weak (dims_direct) baseline lands ~0.1 at seed 42.
"""
from mlsbench.scoring.dsl import *

term("ap25_easy",
    col("ap25_easy").higher().id().sigmoid(ref=const(0.327007), scale=0.0033219181))
term("ap25_moderate",
    col("ap25_moderate").higher().id().sigmoid(ref=const(0.173765), scale=0.0056858093))
term("ap25_hard",
    col("ap25_hard").higher().id().sigmoid(ref=const(0.168744), scale=0.0113452217))

setting("easy", weighted_mean(("ap25_easy", 1.0)))
setting("moderate", weighted_mean(("ap25_moderate", 1.0)))
setting("hard", weighted_mean(("ap25_hard", 1.0)))

task(gmean("easy", "moderate", "hard"))
