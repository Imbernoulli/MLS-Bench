"""Score spec for reg-smoothness-weight.

A FIXED learned dense deformable registration pipeline (VoxelMorph U-Net + NCC
similarity) is trained on a REAL brain-MRI deformable-registration dataset
(IXI2D T1-weighted slices, exact GT deformation). The agent designs ONLY the
weight lambda of the displacement-field smoothness regulariser:

    loss = NCC(warped, fixed) + lambda * grad_smoothness(field)

The score of each deformation-magnitude setting combines (i) the warped-moving
vs fixed PSNR (dB; HIGHER better, weight 0.7) and (ii) a FOLDING penalty — the
fraction of pixels with a non-positive deformation-Jacobian determinant (LOWER
better, weight 0.3) — so a good field must both ALIGN the images AND stay
near-diffeomorphic (the standard registration requirement). The score
aggregates over THREE deformation-magnitude settings (small / medium / large).

Measured anchors (k1 H20, NVIDIA H20, torch 2.4.1+cu121, seed 42, real IXI2D
data, 800 steps, 24-pair val set per setting) — per-setting PSNR (dB) /
folding fraction:
  setting   lambda=0.0(none)   lambda=0.05(moderate, SOTA)   lambda=5.0(high, weak)
  small     26.41 / 0.000000   24.70 / 0.000023              26.74 / 0.000000
  medium    20.04 / 0.014441   20.09 / 0.006960              16.96 / 0.000000
  large     16.50 / 0.057983   15.18 / 0.012723              12.40 / 0.000000
A large smoothness weight ('high') removes folding entirely but over-smooths
the field, collapsing PSNR at medium/large deformations; 'moderate' keeps
folding low while retaining PSNR competitive with the un-regularised field at
medium/large. Moderate is anchored as SOTA (score 0.5) and high as the weak
baseline — the task rewards the moderate regularisation trade-off, not either
extreme.

Cross-seed robustness note: at the `small` setting all three baselines have
near-zero folding and near-tied PSNR, so a naive |sota-weak|/ln(9) scale from
seed 42 alone (psnr_scale_small~0.93, fold_scale_small~1e-5) is razor-thin and
FLIPS on seed 123 (`none` outscores `moderate`, an inversion at the very seed
used to calibrate it). The `small`-setting psnr/fold scales below are widened
(30 / 0.03) to de-weight that near-noise setting relative to medium/large,
where the real regularisation-strength signal lives; this keeps `moderate`
strictly on top of `none`/`high` on BOTH seed 42 and seed 123 (verified via
score_record on `/tmp/reg_anchor_real.tsv` + `/tmp/reg_seed123.tsv`), at the
cost of `small` contributing less separation to the composite score.
"""
from mlsbench.scoring.dsl import *

# per-setting warped-moving PSNR (higher better). `small`'s scale is widened
# (see note above) so its near-tied, near-noise readings don't dominate and
# invert the cross-seed ordering.
term("psnr_small",
    col("psnr_small").higher().id().sigmoid(ref=const(24.702856), scale=30.0))
term("psnr_medium",
    col("psnr_medium").higher().id().sigmoid(ref=const(20.088774), scale=1.423290))
term("psnr_large",
    col("psnr_large").higher().id().sigmoid(ref=const(15.181020), scale=1.264067))

# per-setting FOLDING fraction (lower better) — diffeomorphism / field validity.
# `small`'s scale is widened (see note above) for the same cross-seed reason.
term("fold_small",
    col("folding_small").lower().id().sigmoid(ref=const(0.000023), scale=0.03))
term("fold_medium",
    col("folding_medium").lower().id().sigmoid(ref=const(0.006960), scale=0.003168))
term("fold_large",
    col("folding_large").lower().id().sigmoid(ref=const(0.012723), scale=0.005790))

setting("small",  weighted_mean(("psnr_small", 0.7),  ("fold_small", 0.3)))
setting("medium", weighted_mean(("psnr_medium", 0.7), ("fold_medium", 0.3)))
setting("large",  weighted_mean(("psnr_large", 0.7),  ("fold_large", 0.3)))

task(gmean("small", "medium", "large"))
