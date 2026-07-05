"""Score spec for mono3d-head-capacity.

Primary metric per setting: AP3D@0.25 = fraction of TEST objects (in this KITTI difficulty tier) whose
predicted 3D box has 3D IoU >= 0.25 with GT (HIGHER better; a degenerate predictor scores ~0).
The task score is the geometric mean of the per-setting AP3D over KITTI's own OFFICIAL
easy/moderate/hard difficulty tiers (bbox-height/occlusion/truncation thresholds), applied
EXCLUSIVELY as a disjoint partition of the held-out REAL KITTI test split.

The design lever is head capacity (depth / width). MEASURED anchors (B0 8xH200, torch 2.4.1,
REAL KITTI 3D Object Detection, CROSS-SEED 42/123, seed-averaged, 1200 steps, AP3D@0.25 per
KITTI official difficulty tier):
  backbone_shallow (WEAK)  easy=0.2606  moderate=0.1315  hard=0.1385
  backbone_deep (STRONG)   easy=0.2763  moderate=0.1417  hard=0.1424
The weak->strong order holds across all three settings and both seeds (deep beats shallow
robustly, though margins on real KITTI are modest, 0.004-0.016 AP3D).

SOTA=0.5 anchor convention (matches stereo-disparity-range): "0.5 is the strongest baseline"
-- ref = the STRONG (backbone_deep) baseline's SEED-42-SPECIFIC value (NOT seed-averaged)
so backbone_deep scores exactly 0.5 at seed 42; scale = (strong_seed42-weak_seed42)/ln(9) so
the weak (backbone_shallow) baseline lands ~0.1 at seed 42.
"""
from mlsbench.scoring.dsl import *

term("ap25_easy",
    col("ap25_easy").higher().id().sigmoid(ref=const(0.290511), scale=0.0049831046))
term("ap25_moderate",
    col("ap25_moderate").higher().id().sigmoid(ref=const(0.149347), scale=0.0056858093))
term("ap25_hard",
    col("ap25_hard").higher().id().sigmoid(ref=const(0.146692), scale=0.0026178480))

setting("easy", weighted_mean(("ap25_easy", 1.0)))
setting("moderate", weighted_mean(("ap25_moderate", 1.0)))
setting("hard", weighted_mean(("ap25_hard", 1.0)))

task(gmean("easy", "moderate", "hard"))
