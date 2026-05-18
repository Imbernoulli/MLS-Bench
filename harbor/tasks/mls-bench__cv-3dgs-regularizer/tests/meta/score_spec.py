"""Score spec for cv-3dgs-regularizer."""
from mlsbench.scoring.dsl import *

term("best_psnr_garden",
    col("best_psnr_garden").higher().id()
    .sigmoid())

term("best_psnr_bicycle",
    col("best_psnr_bicycle").higher().id()
    .sigmoid())

term("best_psnr_bonsai",
    col("best_psnr_bonsai").higher().id()
    .sigmoid())

term("best_psnr_stump",
    col("best_psnr_stump").higher().id()
    .sigmoid())

setting("garden", weighted_mean(("best_psnr_garden", 1.0)))
setting("bicycle", weighted_mean(("best_psnr_bicycle", 1.0)))
setting("bonsai", weighted_mean(("best_psnr_bonsai", 1.0)))
setting("stump", weighted_mean(("best_psnr_stump", 1.0)))

task(gmean("garden", "bicycle", "bonsai", "stump"))
