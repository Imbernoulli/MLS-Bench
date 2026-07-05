"""Score spec for stereo-agg-dilation (3 difficulty settings).

Validation disparity end-point error (EPE, pixels; LOWER is better) of a FIXED
small GC-Net/PSMNet-style stereo net trained on REAL rectified stereo
photographs from the Middlebury Stereo Datasets 2005/2006 (structured-light
ground-truth disparity; see vendor/data_scripts/stereo-matching/
prepare_data.py). The agent designs ONLY the spatial dilation rate of the
(fixed) 3D cost-aggregation network's middle conv; every other axis is FIXED.
3 severities vary the scene's disparity range (easy up to ~59px, medium ~70px,
hard ~77px real ground-truth disparity).

GPU re-anchored on real data (k1 H20, NVIDIA H20, torch 2.4.0, seed 42,
package-default 1200 steps): medium was near-tied (d1 0.774 vs d2 0.772) and
easy/hard held weak(d1)>strong(d2). A 3000-step training-budget diagnostic
(same methodology used to resolve `aggregation`/`refine`) was run cross-seed
(42/123) to check robustness:

  3000 steps, seed 42  : easy   d1 3.256 > d2 3.151   (OK)
                          medium d1 11.738 > d2 9.502  (OK, now a clear gap)
                          hard   d1 3.874 > d2 3.631   (OK)
  3000 steps, seed 123 : easy   d1 2.953 < d2 2.989   (FAIL -- tiny, sign-flipped)
                          medium d1 10.472 > d2 8.794  (OK)
                          hard   d1 4.346 > d2 3.407   (OK)

At the longer 3000-step schedule, medium and hard are cleanly monotone
weak(d1)>strong(d2) on BOTH seeds with a real, non-trivial gap -- a wider
receptive field genuinely helps resolve the larger, more ambiguous disparities
of the harder settings once the aggregation network is given enough training
to exploit it. `easy` is a near-tied/noise-level setting on both seeds (gaps
of ~0.1px on a ~3.0-3.9px EPE baseline, roughly 3% either direction) and flips
sign between seeds -- this is floor/noise territory, not a real per-seed
ordering, so `easy`'s scale is floored wide (0.35) rather than using
gap/ln(9), to avoid over-crediting sampling noise on that setting while still
preserving a mild, correctly-signed nudge from the seed-42 numbers used to
calibrate ref/scale for medium/hard.

Ships at the LONGER schedule (3000 steps, not the package default 1200 steps)
-- see scripts/*.sh. Per-setting ref = strong (d2) at seed-42, 3000-step
numbers, so d2 anchors score 0.5; scale = (weak-strong)/ln(9) on medium/hard
so d1 anchors score ~0.1; easy uses a wide floored scale given the near-tied/
sign-flipping real-data behavior.
"""
from mlsbench.scoring.dsl import *

term("epe_easy",
    col("epe_easy").lower().id()
    .sigmoid(ref=const(3.150783), scale=0.35))

term("epe_medium",
    col("epe_medium").lower().id()
    .sigmoid(ref=const(9.502220), scale=1.017454))

term("epe_hard",
    col("epe_hard").lower().id()
    .sigmoid(ref=const(3.630593), scale=0.110853))

setting("easy", weighted_mean(("epe_easy", 1.0)))
setting("medium", weighted_mean(("epe_medium", 1.0)))
setting("hard", weighted_mean(("epe_hard", 1.0)))

task(gmean("easy", "medium", "hard"))
