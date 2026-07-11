"""Score spec for gp-kernel-design (strict-bar 3-dataset eval).

Held-out test negative-log-likelihood (per point, original y scale, LOWER is
better) of an ExactGP whose covariance + mean the agent designs, trained under a
FIXED Type-II MLE pipeline, scored on THREE fixed UCI regression splits
(concrete 1030x8, kin8nm 8192x8, elevators 16.6k x18). The task score is the
geometric mean of the three per-dataset logistic terms; each term's midpoint is
anchored between the isotropic-RBF weak baseline and a well-designed ARD kernel.

Measured anchors (200 Adam iters lr=0.1, seed 42, H20, 2026-07-11):
  dataset    isotropic-RBF   ARD-Matern52
  concrete    3.114257        2.993942
  kin8nm     -0.004929       -0.507105
  elevators  -3.019634       -3.305446

Baseline order (test NLL, lower better), CONFIRMED on all 3 datasets:
  isotropic RBF  >  ARD-Matern-5/2       (ARD is the dominant lever)
Only complete three-setting measurements are admitted as declared baselines.
The prior spectral-mixture attempt did not finish `elevators`, so it is not an
anchor and is excluded instead of being assigned a partial score. NLL is the
primary metric; RMSE is reported for feedback but not scored.

Per-setting logistic midpoint = the measured ARD-Matern-5/2 NLL, which maps to
0.5. Scale = (isotropic-RBF - ARD-Matern-5/2) / ln(9), so the measured native
RBF maps to 0.1. Both complete rows were rerun from the same immutable source
archive and split checksums; partial or failed rows are not calibration input.
"""
from mlsbench.scoring.dsl import *

term("nll_concrete",
    col("nll_concrete").lower().id().sigmoid(ref=const(2.993942), scale=0.054757716275803843))
term("nll_kin8nm",
    col("nll_kin8nm").lower().id().sigmoid(ref=const(-0.507105), scale=0.22855014693527936))
term("nll_elevators",
    col("nll_elevators").lower().id().sigmoid(ref=const(-3.305446), scale=0.1300786469203348))

setting("concrete", weighted_mean(("nll_concrete", 1.0)))
setting("kin8nm", weighted_mean(("nll_kin8nm", 1.0)))
setting("elevators", weighted_mean(("nll_elevators", 1.0)))

task(gmean("concrete", "kin8nm", "elevators"))
