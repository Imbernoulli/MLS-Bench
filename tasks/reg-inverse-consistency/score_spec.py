"""Score spec for reg-inverse-consistency.

A learned dense deformable registration field (VoxelMorph U-Net) is trained on a
REAL brain-MRI deformable-registration dataset (IXI2D T1-weighted slices, exact
GT deformation). The agent designs ONLY the WEIGHT of an inverse / cycle-
consistency term: in addition to the forward field (moving->fixed) the harness
predicts the SWAPPED field (fixed->moving) and penalises the residual of
composing the two (forward-then-reverse should be ~identity), encouraging a
SYMMETRIC / invertible registration (inverse-consistency / SYMNet, Zhang 2018 /
Mok & Chung 2020). This is a REGULARISATION-STRENGTH knob: inverse_w=0 leaves
the field free to align at maximum PSNR; larger weights drive folding to zero
but SACRIFICE alignment; an over-strong weight collapses the deformation. The
score of each setting combines the warped-moving PSNR (dB; HIGHER, weight 0.7)
and a FOLDING penalty (LOWER, weight 0.3). Aggregates over THREE deformation-
magnitude settings.

Measured anchors (k1 H20, NVIDIA H20, torch 2.4.1+cu121, seed 42, real IXI2D
data, 800 steps, 24-pair val set per setting) — per-setting PSNR (dB) /
folding fraction:
  setting   over(w=50, weak)   on(w=1.0)         off(w=0, SOTA)
  small     20.66 / 0.000000   25.87 / 0.000000  25.55 / 0.000021
  medium    15.39 / 0.000000   18.34 / 0.000000  20.11 / 0.006283
  large     12.42 / 0.000000   13.57 / 0.000000  15.10 / 0.012113
On this real MRI deformation, the plain one-directional field (off) already
attains the best PSNR at every setting (folding stays low, single-digit
per-mille); adding cycle-consistency (on) removes the residual folding entirely
but trades away meaningful alignment (PSNR drops several dB), and an
over-strong weight (over) collapses the deformation further. Task-level order
over < on < off holds cleanly; off is anchored as SOTA (score 0.5) and over as
the weak baseline.

Cross-seed robustness note: at the `small` setting folding is near-zero for
all three baselines and PSNR is close between off/on, so a naive
|sota-weak|/ln(9) scale from seed 42 alone (psnr_scale_small~2.22,
fold_scale_small~1e-5) is thin enough to FLIP on seed 123 (`on` outscores
`off`, an inversion at the very seed used to calibrate it). The `small`-setting
psnr/fold scales below are widened (20 / 0.03) to de-weight that near-noise
setting relative to medium/large, where the real regularisation-strength
signal lives; this keeps `off` strictly on top of `on`/`over` on BOTH seed 42
and seed 123 (verified via score_record on `/tmp/reg_anchor_real.tsv` +
`/tmp/reg_seed123.tsv`), at the cost of `small` contributing less separation to
the composite score.
"""
from mlsbench.scoring.dsl import *

# per-setting warped-moving PSNR (higher better). `small`'s scale is widened
# (see note above) so its near-tied, near-noise readings don't dominate and
# invert the cross-seed ordering.
term("psnr_small",
    col("psnr_small").higher().id().sigmoid(ref=const(25.545118), scale=20.0))
term("psnr_medium",
    col("psnr_medium").higher().id().sigmoid(ref=const(20.107546), scale=2.147615))
term("psnr_large",
    col("psnr_large").higher().id().sigmoid(ref=const(15.100230), scale=1.221320))

# per-setting FOLDING fraction (lower better) — diffeomorphism / field validity.
# `small`'s scale is widened (see note above) for the same cross-seed reason.
term("fold_small",
    col("folding_small").lower().id().sigmoid(ref=const(0.000021), scale=0.03))
term("fold_medium",
    col("folding_medium").lower().id().sigmoid(ref=const(0.006283), scale=0.002860))
term("fold_large",
    col("folding_large").lower().id().sigmoid(ref=const(0.012113), scale=0.005513))

setting("small",  weighted_mean(("psnr_small", 0.7),  ("fold_small", 0.3)))
setting("medium", weighted_mean(("psnr_medium", 0.7), ("fold_medium", 0.3)))
setting("large",  weighted_mean(("psnr_large", 0.7),  ("fold_large", 0.3)))

task(gmean("small", "medium", "large"))
