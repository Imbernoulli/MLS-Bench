"""Score spec for stereo-disparity-refinement (3 difficulty settings).

Validation disparity end-point error (EPE, pixels; LOWER is better) of a FIXED
small GC-Net/PSMNet-style stereo net trained for a LONGER schedule (3000
steps, not the package default 1200 steps -- see scripts/*.sh) on REAL
rectified stereo photographs from the Middlebury Stereo Datasets 2005/2006
(structured-light ground-truth disparity; see
vendor/data_scripts/stereo-matching/prepare_data.py). The agent designs ONLY
the post-hoc, left-image-guided disparity refinement (none vs a small
residual refinement CNN, cf. StereoNet/iResNet edge-aware refinement); every
other axis is FIXED. 3 severities vary the scene's disparity range (easy up
to ~59px, medium ~70px, hard ~77px real ground-truth disparity).

GPU re-anchored on real data (k1 H20, NVIDIA H20, torch 2.4.0). At the
package's default 1200 steps the none/residual order was INVERTED on
easy+hard (undertraining artifact -- the refinement head needs the base
disparity map to already be reasonably converged before its edge-aware
correction helps rather than adds noise). A diagnostic re-run at 3000 steps
resolves this cleanly on medium+hard on seed 42 and is CONFIRMED on a second
seed (123); easy remains a near-tied setting (tiny/noisy gap, sign flips
between seeds) with no real headroom to separate the two configs there:

  seed 42  (3000 steps): easy   none 3.007 ~= residual 2.918 (near-tied)
                          medium none 10.769 > residual 9.259
                          hard   none 4.308 > residual 4.062
  seed 123 (3000 steps): easy   none 2.950 ~= residual 2.978 (near-tied, tiny
                                 sign flip -- noise level, not a real reversal)
                          medium none 11.599 > residual 9.299
                          hard   none 5.820 > residual 3.699

The residual refinement head conditions on the raw left image to correct the
initial disparity map (edge-aware smoothing, sharpening depth discontinuities);
it wins clearly on medium+hard once training runs long enough (3000 steps);
easy is near the metric floor on this FIXED small net (both configs already
converge, tiny/noisy gap), so its scale is floored wider than gap/ln(9) would
give to avoid over-crediting sampling noise. Per-setting ref = strong
(residual) at seed 42/3000-step numbers, so residual anchors score 0.5; scale
= (weak-strong)/ln(9) so none anchors score ~0.1 (floored on easy).
"""
from mlsbench.scoring.dsl import *

term("epe_easy",
    col("epe_easy").lower().id()
    .sigmoid(ref=const(2.918357), scale=0.05))

term("epe_medium",
    col("epe_medium").lower().id()
    .sigmoid(ref=const(9.259033), scale=0.687017))

term("epe_hard",
    col("epe_hard").lower().id()
    .sigmoid(ref=const(4.061729), scale=0.112144))

setting("easy", weighted_mean(("epe_easy", 1.0)))
setting("medium", weighted_mean(("epe_medium", 1.0)))
setting("hard", weighted_mean(("epe_hard", 1.0)))

task(gmean("easy", "medium", "hard"))
