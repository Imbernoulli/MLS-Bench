"""Score spec for reg-diffeomorphic-integration.

A learned dense deformable registration field (VoxelMorph U-Net) is trained on a
REAL brain-MRI deformable-registration dataset (IXI2D T1-weighted slices, exact
GT deformation). The agent designs ONLY the number of SCALING-AND-SQUARING
integration steps that turn the network's output into the final displacement
(0 = plain displacement that can FOLD; 5-7 = integrate a stationary velocity
field into a fold-free DIFFEOMORPHISM, VoxelMorph-diff / Dalca et al. 2018). The
score of each setting combines the warped-moving vs fixed PSNR (dB; HIGHER
better, weight 0.7) and a FOLDING penalty — fraction of pixels with
non-positive Jacobian determinant (LOWER better, weight 0.3) — so a good
transform must ALIGN and stay diffeomorphic. Aggregates over THREE
deformation-magnitude settings.

Measured anchors (k1 H20, NVIDIA H20, torch 2.4.1+cu121, seed 42, real IXI2D
data, 800 steps, 24-pair val set per setting) — per-setting PSNR (dB) /
folding fraction:
  setting   none(steps=0, weak)   light(steps=3)   full(steps=7, SOTA)
  small     26.79 / 0.000000      25.29 / 0.000000  25.97 / 0.000000
  medium    21.06 / 0.006505      20.85 / 0.000000  20.95 / 0.000000
  large     15.46 / 0.012979      17.14 / 0.000000  15.74 / 0.000000
Without integration the field develops folds (non-positive Jacobian) that hurt
validity, worst at medium/large deformations; full integration removes the
folding entirely (folding=0 at every setting) while keeping alignment
competitive. Scoring rewards the fold-free `full` baseline as SOTA (score
~0.5); `none`'s residual folding pulls it below at medium/large despite a
higher raw small-setting PSNR. Per-setting anchors below (SOTA -> score 0.5,
scale = |sota - weak| / ln(9)).
"""
from mlsbench.scoring.dsl import *

# per-setting warped-moving PSNR (higher better)
term("psnr_small",
    col("psnr_small").higher().id().sigmoid(ref=const(25.971562), scale=0.372529))
term("psnr_medium",
    col("psnr_medium").higher().id().sigmoid(ref=const(20.951629), scale=0.049085))
term("psnr_large",
    col("psnr_large").higher().id().sigmoid(ref=const(15.742421), scale=0.127542))

# per-setting FOLDING fraction (lower better) — diffeomorphism / field validity
term("fold_small",
    col("folding_small").lower().id().sigmoid(ref=const(0.0), scale=0.004551))
term("fold_medium",
    col("folding_medium").lower().id().sigmoid(ref=const(0.0), scale=0.002961))
term("fold_large",
    col("folding_large").lower().id().sigmoid(ref=const(0.0), scale=0.005907))

setting("small",  weighted_mean(("psnr_small", 0.7),  ("fold_small", 0.3)))
setting("medium", weighted_mean(("psnr_medium", 0.7), ("fold_medium", 0.3)))
setting("large",  weighted_mean(("psnr_large", 0.7),  ("fold_large", 0.3)))

task(gmean("small", "medium", "large"))
