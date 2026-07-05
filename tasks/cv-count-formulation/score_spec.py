"""Score spec for cv-count-formulation.

Counting MAE (lower is better) of a crowd-counting head (fixed CSRNet-style frontend,
agent-editable HEAD only: a bare scalar-count regressor vs a density-map head whose
spatial integral gives the count) trained on REAL ShanghaiTech-derived crowd scenes
(Zhang et al., CVPR 2016), scored on THREE density scenes (medium / middense / dense)
each a count-EXTRAPOLATION test (train counts LOW, held-out val counts HIGHER, disjoint
ranges) so a degenerate constant-mean predictor cannot win by construction. Score is the
geometric mean over the three scenes.

Fresh GPU re-anchor on REAL ShanghaiTech data (k1 H20, torch 2.4.0, cross-seed 42/123,
vendor/crowd-counting/run_anchors.py -> anchor_real_full.tsv). At the ORIGINAL
package-default budget (300 iters) this surface was NOT cross-seed clean (middense/dense
sign-flipped on at least one seed). A training-budget diagnostic at 1500 iters (5x
longer, same seeds, NOT an HP-sweep -- a single independent longer-budget rerun) resolved
this: density (strong) beats scalar (weak) on EVERY scene and BOTH seeds individually at
1500 iters, so this was an undertraining artifact, not a genuine confound. Shipping at
the validated 1500-iter budget (scripts/count_*.sh COUNT_ITERS default updated 300->1500):

  medium   seed42  weak=58.2806 strong=35.5993  seed123 weak=53.9109 strong=50.5953
  middense seed42  weak=68.1507 strong=47.4239  seed123 weak=70.1250 strong=52.7991
  dense    seed42  weak=93.2756 strong=68.2459  seed123 weak=84.1387 strong=72.1466

Anchors below use the SEED-42 values (matching the seed-42 leaderboard.csv rows that get
scored), so `baseline:density` (strong) scores exactly ref_score=0.5 and `baseline:scalar`
(weak) scores ~0.1 -- the seed-123 numbers above are cross-seed provenance confirming the
ordering is robust, not the pinned anchor. Per-scene ref = strong (density) seed-42 value;
scale = (weak-strong)/ln(9) so weak (scalar) scores ~0.1.
"""
from mlsbench.scoring.dsl import *

_STRONG = {"medium": 35.5993, "middense": 47.4239, "dense": 68.2459}
_SCALE = {"medium": 10.322704, "middense": 9.433173, "dense": 11.391507}

for _s in ("medium", "middense", "dense"):
    term(f"mae_{_s}",
        col(f"mae_{_s}").lower().id().sigmoid(ref=const(_STRONG[_s]), scale=_SCALE[_s]))
    setting(_s, weighted_mean((f"mae_{_s}", 1.0)))

task(gmean("medium", "middense", "dense"))
