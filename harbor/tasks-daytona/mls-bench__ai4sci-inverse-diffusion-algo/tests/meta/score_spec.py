"""Score spec for ai4sci-inverse-diffusion-algo.

Diffusion-based inverse problem solving across three settings:

1. inv-scatter (inverse scattering, optical tomography):
   - psnr: higher is better, unbounded above, sigmoid
   - ssim: higher is better, bounded in [0, 1], bounded_power

2. blackhole (black hole imaging, EHT):
   - psnr: higher is better, unbounded above, sigmoid (the only objective)
   - cp_chi2, camp_chi2: data-consistency constraints (reduced chi^2 <= 2),
     multiplicative penalty_upper. See the comment at the blackhole block.

3. inpainting (FFHQ256 face image inpainting):
   - psnr: higher is better, unbounded above, sigmoid
   - ssim: higher is better, bounded in [0, 1], bounded_power
   - lpips: lower is better, bounded at 0, bounded_power

Baselines: dps (score-based guidance), reddiff (optimization-based), lgd (MC guidance).

Best baselines (single seed=42, observed on leaderboard):
  inv-scatter: reddiff psnr=38.25, ssim=0.982
  blackhole:   reddiff psnr=21.75, cp_chi2=3.57, camp_chi2=3.49
  inpainting:  reddiff psnr=22.11, ssim=0.751, lpips=0.163

ref values are set near the strongest baseline so that baseline scores ~0.5.
"""
from mlsbench.scoring.dsl import *

# ---- inv-scatter ----
term("psnr_scatter",
    col("psnr_inv-scatter").higher().id()
    .sigmoid())
term("ssim_scatter",
    col("ssim_inv-scatter").higher().id()
    .bounded_power(bound=1.0))

setting("inv-scatter", weighted_mean(
    ("psnr_scatter", 1.0), ("ssim_scatter", 1.0)))

# ---- blackhole ----
# The closure-phase / log-closure-amplitude reduced chi^2 is a data-consistency
# check, not an image-quality score: InverseBench's evaluate_chisq already folds
# chi^2 < 1 to 1/chi^2, so the reported value is >= 1 and the ideal is exactly 1
# (fit to within the noise), which a prior-free fit of forward_op.loss reaches
# without producing a good image. As two thirds of an averaged objective (with
# bound=0, a value the metric cannot take) it paid such a fit more than every
# baseline. Image fidelity (PSNR vs. ground truth) is the objective; chi^2 above
# 2 multiplies the setting by exp(-0.15 * (chi^2 - 2)) per closure quantity.
term("psnr_bh",
    col("psnr_blackhole").higher().id()
    .sigmoid())
term("cp_chi2_bh",
    penalty_upper(col("cp_chi2_blackhole").lower().id(), target=2.0, sharpness=0.15))
term("camp_chi2_bh",
    penalty_upper(col("camp_chi2_blackhole").lower().id(), target=2.0, sharpness=0.15))

setting("blackhole", weighted_mean(("psnr_bh", 1.0)),
    constraints=["cp_chi2_bh", "camp_chi2_bh"])

# ---- inpainting ----
term("psnr_inpaint",
    col("psnr_inpainting").higher().id()
    .sigmoid())
term("ssim_inpaint",
    col("ssim_inpainting").higher().id()
    .bounded_power(bound=1.0))
term("lpips_inpaint",
    col("lpips_inpainting").lower().id()
    .bounded_power(bound=0.0))

setting("inpainting", weighted_mean(
    ("psnr_inpaint", 1.0), ("ssim_inpaint", 1.0), ("lpips_inpaint", 1.0)))

# Task: geometric mean across the three problem types
task(gmean("inv-scatter", "blackhole", "inpainting"))
