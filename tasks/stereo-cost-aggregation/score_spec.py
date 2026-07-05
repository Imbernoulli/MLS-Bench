"""Score spec for stereo-cost-aggregation (PRIMARY strict-bar surface, 3 settings).

Validation disparity end-point error (EPE, pixels; LOWER is better) of a stereo
net trained for a LONGER schedule (3000 steps, not the package default 1200
steps -- see scripts/*.sh) on REAL rectified stereo photographs from the
Middlebury Stereo Datasets 2005/2006 (structured-light ground-truth disparity;
see vendor/data_scripts/stereo-matching/prepare_data.py). The agent designs
ONLY the cost-AGGREGATION architecture; features, cost-volume kind, disparity
readout, loss and schedule are FIXED. 3 severities vary the scene's disparity
range (easy up to ~59px, medium ~70px, hard ~77px real ground-truth
disparity).

GPU re-anchored on real data (k1 H20, NVIDIA H20, torch 2.4.0). At the
package's default 1200 steps the bm/conv2d/conv3d order was INVERTED on the
easy+medium settings (undertraining artifact -- classical block-matching's
smoother, lower-variance output looks deceptively better before the deep
aggregation nets have converged). A diagnostic re-run at 3000 steps fully
resolves this on seed 42 AND is CONFIRMED on a second seed (123) -- STRICT
ORDER bm > conv2d > conv3d (EPE) holds on ALL 3 settings, both seeds:

  seed 42  (3000 steps): easy   bm 5.829 > conv2d 3.317 > conv3d 3.040
                          medium bm 11.302 > conv2d 10.252 > conv3d 9.313
                          hard   bm 8.719 > conv2d 4.852 > conv3d 3.896
  seed 123 (3000 steps): easy   bm 6.075 > conv2d 3.267 > conv3d 3.054
                          medium bm 11.532 > conv2d 11.277 > conv3d 9.362
                          hard   bm 8.589 > conv2d 5.488 > conv3d 3.295

Classical non-learned windowed SAD block-matching (bm, Scharstein & Szeliski,
IJCV 2002) has no cost-volume regression network at all; per-slice-independent
2D-conv aggregation (conv2d, PSMNet ablation) learns SOME structure but no
cross-disparity context; the full 3D-conv cost-volume aggregation (conv3d,
GC-Net/PSMNet, Chang & Chen CVPR 2018) is SOTA -- reproducing the literature
order bm < conv2d < conv3d (in accuracy) on every setting, once trained long
enough. Per-setting ref = strong (conv3d) at seed 42/3000-step numbers, so
conv3d anchors score 0.5; scale = (weak-strong)/ln(9) so bm anchors score ~0.1.
"""
from mlsbench.scoring.dsl import *

term("epe_easy",
    col("epe_easy").lower().id()
    .sigmoid(ref=const(3.039525), scale=1.269338))

term("epe_medium",
    col("epe_medium").lower().id()
    .sigmoid(ref=const(9.313473), scale=0.905108))

term("epe_hard",
    col("epe_hard").lower().id()
    .sigmoid(ref=const(3.896312), scale=2.194862))

setting("easy", weighted_mean(("epe_easy", 1.0)))
setting("medium", weighted_mean(("epe_medium", 1.0)))
setting("hard", weighted_mean(("epe_hard", 1.0)))

task(gmean("easy", "medium", "hard"))
