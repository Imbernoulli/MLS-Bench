"""Score spec for cv-count-normalization.

Counting MAE (lower is better) of a crowd-counting density-map head's spatial
NORMALIZATION (fixed CSRNet-style frontend + density head, agent-editable normalization
only: softmax-normalized density map vs a free/unnormalized density map) trained on REAL
ShanghaiTech-derived crowd scenes (Zhang et al., CVPR 2016), scored on THREE density
scenes (medium / middense / dense) each a count-EXTRAPOLATION test (train counts LOW,
held-out val counts HIGHER, disjoint ranges) so a degenerate constant-mean predictor
cannot win by construction. Score is the geometric mean over the three scenes.

Fresh GPU re-anchor on REAL ShanghaiTech data (k1 H20, torch 2.4.0, cross-seed 42/123,
vendor/crowd-counting/run_anchors.py -> anchor_real_full.tsv). At the ORIGINAL
package-default budget (300 iters) this surface was NOT cross-seed clean (medium/middense
sign-flipped on at least one seed). A training-budget diagnostic at 1500 iters (5x
longer, same seeds, NOT an HP-sweep -- a single independent longer-budget rerun) resolved
this: free (strong) beats softmax (weak) on EVERY scene and BOTH seeds individually at
1500 iters, so this was an undertraining artifact, not a genuine confound. Shipping at
the validated 1500-iter budget (scripts/count_*.sh COUNT_ITERS default updated 300->1500):

  medium   seed42  weak=50.3713 strong=42.4631  seed123 weak=50.3474 strong=45.9450
  middense seed42  weak=80.4350 strong=49.2733  seed123 weak=80.4058 strong=56.4417
  dense    seed42  weak=119.1146 strong=73.5902  seed123 weak=119.0776 strong=71.8760

Anchors below use the SEED-42 values (matching the seed-42 leaderboard.csv rows that get
scored), so `baseline:free` (strong) scores exactly ref_score=0.5 and `baseline:softmax`
(weak) scores ~0.1 -- the seed-123 numbers above are cross-seed provenance confirming the
ordering is robust, not the pinned anchor. Per-scene ref = strong (free) seed-42 value;
scale = (weak-strong)/ln(9) so weak (softmax) scores ~0.1.
"""
from mlsbench.scoring.dsl import *

_STRONG = {"medium": 42.4631, "middense": 49.2733, "dense": 73.5902}
_SCALE = {"medium": 3.599177, "middense": 14.182301, "dense": 20.719047}

for _s in ("medium", "middense", "dense"):
    term(f"mae_{_s}",
        col(f"mae_{_s}").lower().id().sigmoid(ref=const(_STRONG[_s]), scale=_SCALE[_s]))
    setting(_s, weighted_mean((f"mae_{_s}", 1.0)))

task(gmean("medium", "middense", "dense"))
