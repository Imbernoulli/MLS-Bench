"""Score spec for reg-regularization-type.

A learned dense deformable registration field (VoxelMorph U-Net + NCC similarity,
fixed regulariser WEIGHT) is trained on a REAL brain-MRI deformable-registration
dataset (IXI2D T1-weighted slices, exact GT deformation). The agent designs ONLY
the TYPE of the displacement-field regulariser: 'none' (no penalty -> the field
folds), 'l2' (first-order diffusion / gradient penalty, VoxelMorph default), or
'bending' (second-order bending-energy / TPS penalty, Rueckert et al. 1999). The
score of each setting combines the warped-moving PSNR (dB; HIGHER, weight 0.7)
and a FOLDING penalty (fraction of non-positive-Jacobian pixels; LOWER, weight
0.3). Aggregates over THREE deformation-magnitude settings.

Measured anchors (k1 H20, NVIDIA H20, torch 2.4.1+cu121, seed 42, real IXI2D
data, 800 steps, 24-pair val set per setting) — per-setting PSNR (dB) /
folding fraction:
  setting   none            l2 (SOTA)        bending (weak)
  small     24.68 / 0.000041  26.11 / 0.000000  24.63 / 0.000023
  medium    20.72 / 0.011827  19.52 / 0.004580  20.31 / 0.011067
  large     16.48 / 0.076097  15.19 / 0.013346  15.22 / 0.040114
On real MRI data 'l2' gives the cleanest, lowest-folding field at small/medium
deformations (best PSNR at small, lowest folding at large) while 'bending'
trails both on PSNR and folding at large. The task-level order none < bending <
l2 holds via the combined PSNR+folding score; l2 is anchored as SOTA (score
0.5) and bending as the weak baseline. Per-setting anchors below (SOTA -> score
0.5, scale = |sota - weak| / ln(9)).
"""
from mlsbench.scoring.dsl import *

# per-setting warped-moving PSNR (higher better)
term("psnr_small",
    col("psnr_small").higher().id().sigmoid(ref=const(26.106786), scale=0.669884))
term("psnr_medium",
    col("psnr_medium").higher().id().sigmoid(ref=const(19.524379), scale=0.357334))
term("psnr_large",
    col("psnr_large").higher().id().sigmoid(ref=const(15.186871), scale=0.015795))

# per-setting FOLDING fraction (lower better) — diffeomorphism / field validity
term("fold_small",
    col("folding_small").lower().id().sigmoid(ref=const(0.0), scale=0.00001))
term("fold_medium",
    col("folding_medium").lower().id().sigmoid(ref=const(0.004580), scale=0.002952))
term("fold_large",
    col("folding_large").lower().id().sigmoid(ref=const(0.013346), scale=0.012183))

setting("small",  weighted_mean(("psnr_small", 0.7),  ("fold_small", 0.3)))
setting("medium", weighted_mean(("psnr_medium", 0.7), ("fold_medium", 0.3)))
setting("large",  weighted_mean(("psnr_large", 0.7),  ("fold_large", 0.3)))

task(gmean("small", "medium", "large"))
