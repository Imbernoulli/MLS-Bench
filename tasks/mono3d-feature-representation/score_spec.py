"""Score spec for mono3d-feature-representation.

Primary metric per setting: AP3D@0.25 = fraction of TEST objects (in this KITTI difficulty tier) whose
predicted 3D box has 3D IoU >= 0.25 with GT (HIGHER better; a degenerate predictor scores ~0).
The task score is the geometric mean of the per-setting AP3D over KITTI's own OFFICIAL
easy/moderate/hard difficulty tiers (bbox-height/occlusion/truncation thresholds), applied
EXCLUSIVELY as a disjoint partition of the held-out REAL KITTI test split.

The design lever is feature representation / fusion. MEASURED anchors (B0 8xH200, torch 2.4.1,
REAL KITTI 3D Object Detection, CROSS-SEED 42/123, seed-averaged, 1200 steps, AP3D@0.25 per
KITTI official difficulty tier):
  feature_appearance_only (WEAK) easy=0.2318  moderate=0.1085  hard=0.1088
  feature_fused (STRONG)         easy=0.3420  moderate=0.1650  hard=0.1783
Fusing the geometry feature vector with the appearance embedding robustly and substantially
beats appearance-only on every tier, both seeds. Per-setting logistic midpoint = the strong
(feature_fused) reference -> score 0.5, scale = (strong-weak)/ln(9) so the weak
(feature_appearance_only) baseline lands ~0.1.
"""
from mlsbench.scoring.dsl import *

term("ap25_easy",
    col("ap25_easy").higher().id().sigmoid(ref=const(0.341971), scale=0.05016305621959836))
term("ap25_moderate",
    col("ap25_moderate").higher().id().sigmoid(ref=const(0.164963), scale=0.02571516839143478))
term("ap25_hard",
    col("ap25_hard").higher().id().sigmoid(ref=const(0.178332), scale=0.03163559188122239))

setting("easy", weighted_mean(("ap25_easy", 1.0)))
setting("moderate", weighted_mean(("ap25_moderate", 1.0)))
setting("hard", weighted_mean(("ap25_hard", 1.0)))

task(gmean("easy", "moderate", "hard"))
