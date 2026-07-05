"""Score spec for stereo-disparity-regression (3 difficulty settings).

Validation disparity end-point error (EPE, pixels; LOWER is better) of a FIXED
small GC-Net-style stereo net trained for a short schedule (1200 steps, seed 42)
on REAL rectified stereo photographs from the Middlebury Stereo Datasets 2005/2006 (structured-light ground-truth disparity; see vendor/data_scripts/stereo-matching/prepare_data.py). The agent designs ONLY the disparity readout from
the aggregated cost volume. 3 severities vary the scene's disparity range (easy up to ~59px, medium ~70px, hard ~77px real ground-truth disparity).

GPU re-anchored on real data (k1 H20, NVIDIA H20, torch 2.4.0, 1200 steps,
20-pair val set) -- CONFIRMED on a second seed (123), STRICT ORDER holds on
ALL 3 settings both seeds:
  seed 42  : easy   argmax 9.373 > softargmin 3.650
             medium argmax 20.451 > softargmin 11.645
             hard   argmax 11.984 > softargmin 5.253
  seed 123 : easy   argmax 8.880 > softargmin 3.086
             medium argmax 20.427 > softargmin 8.920
             hard   argmax 12.129 > softargmin 4.011
The non-differentiable, integer winner-take-all argmax readout never trains
meaningfully (no gradient to the cost volume) and its error grows with the
scene's disparity range; the differentiable sub-pixel soft-argmin learns real
disparity and wins on ALL 3 settings, both seeds. Per-setting ref = strong
(softargmin) at seed-42 numbers, so softargmin anchors score 0.5; scale =
(weak-strong)/ln(9) so argmax anchors score ~0.1.
"""
from mlsbench.scoring.dsl import *

term("epe_easy",
    col("epe_easy").lower().id()
    .sigmoid(ref=const(3.649696), scale=2.604981))

term("epe_medium",
    col("epe_medium").lower().id()
    .sigmoid(ref=const(11.644566), scale=4.008029))

term("epe_hard",
    col("epe_hard").lower().id()
    .sigmoid(ref=const(5.253356), scale=3.063379))

setting("easy", weighted_mean(("epe_easy", 1.0)))
setting("medium", weighted_mean(("epe_medium", 1.0)))
setting("hard", weighted_mean(("epe_hard", 1.0)))

task(gmean("easy", "medium", "hard"))
