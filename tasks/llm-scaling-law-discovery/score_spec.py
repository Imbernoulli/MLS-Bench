"""Score spec for llm-scaling-law-discovery.

Scaling law discovery task: predict LLM performance from compute/data/model
parameters. Three harder dataset settings recommended by the SLDBench
authors: sld-vocab, sld-lrbsz, sld-dataconstrained.

The parser reports four metrics per setting (r2, mae, rmse, nmae), all computed
on the setting's fixed held-out test set. Only two of them are independent:
  - r2   = 1 - rmse^2 / Var(y_test)   -> a monotone function of rmse
  - nmae = mae / Std(y_test)          -> a constant multiple of mae
Each setting is therefore scored on two terms, keeping the original 3:2 split
between the squared-error family (r2 weight 2 + rmse weight 1) and the
absolute-error family (mae weight 1 + nmae weight 1):

  - fvu: log of the fraction of variance unexplained, 1 - r2. Since
    log(1 - r2) = 2*log(rmse) - log(Var(y_test)) and the sigmoid calibration
    below is invariant to that affine change, the term reads the rmse column
    with a log transform and scores exactly as a sigmoid of -log(1 - r2) would.
  - mae: log mae (equivalent to log nmae for the same reason).

Both terms use a sigmoid in log-error space, floored at the worst baseline and
calibrated so the best baseline scores 0.5; a perfect fit approaches 1.0.

Why log space: the kernel_ridge baseline is an outlier (r2 = -413.7 on
sld-lrbsz, -13.4 on sld-dataconstrained). Raw r2 with bounded_power(bound=1)
put every honest fit in the top 0.5 % of the [floor, 1] interval, so all three
r2 terms fell into the pathological-ref sigmoid fallback and scored ~0.5 for
any r2 in [-1, 1] (a perfect fit reached only ~0.505). The linear-space
mae/rmse terms needed gamma > 10 on sld-lrbsz and were clamped. In log space
the outlier is a factor ~20 in rmse rather than ~400 in r2, and relative error
improvements keep their resolution near r2 = 1.
"""
from mlsbench.scoring.dsl import *

# ---- sld-vocab ----
term("fvu_vocab",
    col("rmse_sld_vocab").lower().log().sigmoid())
term("mae_vocab",
    col("mae_sld_vocab").lower().log().sigmoid())

setting("sld-vocab", weighted_mean(
    ("fvu_vocab", 3.0),
    ("mae_vocab", 2.0),
))

# ---- sld-lrbsz ----
term("fvu_lrbsz",
    col("rmse_sld_lrbsz").lower().log().sigmoid())
term("mae_lrbsz",
    col("mae_sld_lrbsz").lower().log().sigmoid())

setting("sld-lrbsz", weighted_mean(
    ("fvu_lrbsz", 3.0),
    ("mae_lrbsz", 2.0),
))

# ---- sld-dataconstrained ----
term("fvu_dataconstrained",
    col("rmse_sld_dataconstrained").lower().log().sigmoid())
term("mae_dataconstrained",
    col("mae_sld_dataconstrained").lower().log().sigmoid())

setting("sld-dataconstrained", weighted_mean(
    ("fvu_dataconstrained", 3.0),
    ("mae_dataconstrained", 2.0),
))

# Task: geometric mean across scaling law datasets
task(gmean("sld-vocab", "sld-lrbsz", "sld-dataconstrained"))
