"""Score spec for stereo-disparity-range (3 difficulty settings: easy/medium/hard).

Validation disparity end-point error (EPE, pixels; LOWER is better) of a FIXED
small GC-Net-style stereo net trained for a short schedule (1200 steps, seed 42)
on REAL rectified stereo photographs (Middlebury Stereo Datasets 2005/2006,
structured-light ground-truth disparity; see
vendor/data_scripts/stereo-matching/prepare_data.py). The agent designs ONLY
the max disparity (D_MAX, number of cost-volume levels); the (real) scene's
disparity range grows with severity (easy up to ~59px, medium ~70px, hard
~77px) so a fixed small D_MAX increasingly clips the true disparities as
severity rises. `full` is D_MAX=96 (large enough to cover the real hard
setting's ~77px); `small` is D_MAX=8.

GPU re-anchored on real data (k1 H20, NVIDIA H20, torch 2.4.0, 1200 steps,
20-pair val set) -- CONFIRMED on a second seed (123), STRICT ORDER holds on
ALL 3 settings both seeds:
  seed 42  : easy   small 9.335 > full 4.612
             medium small 13.384 > full 11.661
             hard   small 13.633 > full 8.283
  seed 123 : easy   small 9.736 > full 3.851
             medium small 13.655 > full 9.341
             hard   small 12.776 > full 7.193
Order small(weak) > full(strong) EPE holds on ALL 3 settings, both seeds
(small increasingly clips as the scene's disparity range grows). Per-setting
ref = strong (full) at seed-42 numbers, so full anchors score 0.5; scale =
(weak-strong)/ln(9) so small anchors score ~0.1.
"""
from mlsbench.scoring.dsl import *

term("epe_easy",
    col("epe_easy").lower().id()
    .sigmoid(ref=const(4.611844), scale=2.149776))

term("epe_medium",
    col("epe_medium").lower().id()
    .sigmoid(ref=const(11.661026), scale=0.784227))

term("epe_hard",
    col("epe_hard").lower().id()
    .sigmoid(ref=const(8.282545), scale=2.435020))

setting("easy", weighted_mean(("epe_easy", 1.0)))
setting("medium", weighted_mean(("epe_medium", 1.0)))
setting("hard", weighted_mean(("epe_hard", 1.0)))

task(gmean("easy", "medium", "hard"))
