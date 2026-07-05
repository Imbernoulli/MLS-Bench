"""Score spec for stereo-agg-normalization (3 difficulty settings).

Validation disparity end-point error (EPE, pixels; LOWER is better) of a FIXED
small GC-Net/PSMNet-style stereo net trained for a short schedule (1200 steps,
seed 42) on REAL rectified stereo photographs from the Middlebury Stereo Datasets 2005/2006 (structured-light ground-truth disparity; see vendor/data_scripts/stereo-matching/prepare_data.py). The agent designs ONLY the
normalization layer inside the (fixed) 3D-conv cost-aggregation network; every
other axis (features, cost volume, readout, loss, schedule) is FIXED. 3
severities vary the scene's disparity range (easy up to ~59px, medium ~70px,
hard ~77px real ground-truth disparity).

GPU re-anchored on real data (k1 H20, NVIDIA H20, torch 2.4.0, 1200 steps,
20-pair val set) -- CONFIRMED on a second seed (123), STRICT ORDER holds on
ALL 3 settings both seeds:
  seed 42  : easy   none 3.660 ~= batch 3.639 (near-tied, tiny gap)
             medium none 13.104 > batch 11.660
             hard   none 10.939 > batch 5.680
  seed 123 : easy   none 3.669 > batch 3.010
             medium none 10.086 > batch 8.818
             hard   none 8.425 > batch 4.189
Without normalization the aggregation network's internal activation statistics
are unconstrained over the short schedule; as the scene's disparity range (and
thus the required cost-volume dynamic range) grows, this destabilises training
badly (hard: EPE nearly doubles) while BatchNorm3d (GC-Net/PSMNet convention)
stays stable -- batch wins clearly on medium+hard on both seeds; easy is near
the metric floor where both configs already converge (seed-42 gap 0.02px,
noise-level; seed-123 gap is real but still small relative to medium/hard), so
scale is floored wider than gap/ln(9) would give to avoid over-crediting
sampling noise. Per-setting ref = strong (batch) at seed-42 numbers, so batch
anchors score 0.5; scale = (weak-strong)/ln(9) so none anchors score ~0.1
(floored on easy).
"""
from mlsbench.scoring.dsl import *

term("epe_easy",
    col("epe_easy").lower().id()
    .sigmoid(ref=const(3.638503), scale=0.05))

term("epe_medium",
    col("epe_medium").lower().id()
    .sigmoid(ref=const(11.659824), scale=0.657080))

term("epe_hard",
    col("epe_hard").lower().id()
    .sigmoid(ref=const(5.680261), scale=2.393496))

setting("easy", weighted_mean(("epe_easy", 1.0)))
setting("medium", weighted_mean(("epe_medium", 1.0)))
setting("hard", weighted_mean(("epe_hard", 1.0)))

task(gmean("easy", "medium", "hard"))
