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
robustly, though margins on real KITTI are modest, 0.004-0.016 AP3D). Per-setting logistic
midpoint = the strong (backbone_deep) reference -> score 0.5, scale = (strong-weak)/ln(9) so
the weak (backbone_shallow) baseline lands ~0.1.
"""
from mlsbench.scoring.dsl import *

term("ap25_easy",
    col("ap25_easy").higher().id().sigmoid(ref=const(0.276278), scale=0.007142419651534148))
term("ap25_moderate",
    col("ap25_moderate").higher().id().sigmoid(ref=const(0.141681), scale=0.004652005127483106))
term("ap25_hard",
    col("ap25_hard").higher().id().sigmoid(ref=const(0.142377), scale=0.0017451561572502997))

setting("easy", weighted_mean(("ap25_easy", 1.0)))
setting("moderate", weighted_mean(("ap25_moderate", 1.0)))
setting("hard", weighted_mean(("ap25_hard", 1.0)))

task(gmean("easy", "moderate", "hard"))
