"""Score spec for cv-harmonization-input-norm.

Foreground-region PSNR (dB, HIGHER is better) of an image harmonizer trained 500 steps
with the agent's design of the 'inputnorm' surface, scored ONLY inside the pasted
foreground region on a HELD-OUT split of REAL iHarmony4 composites (mild=HCOCO,
medium=Hday2night, strong=HFlickr). Every other design axis is FIXED at the strong
reference config, so any change in the score is attributable to this surface.
Weak = background-whitening (naive, degrading) ('bg_whiten'); strong = raw composite input ('none').

Anchors re-measured this session (k1 H20, torch 2.4.0+cu121, 500 iters, cross-seed
42/123 GPU sweep; both seeds confirm strong > weak on EVERY severity -- see
vendor/image-harmonization/run_anchors.py + anchor_real_full.tsv):
  mild    : bg_whiten  11.18 | none       18.20
  medium  : bg_whiten  10.92 | none       19.65
  strong  : bg_whiten  10.94 | none       18.28

Sigmoid convention (SOTA=0.5): ref=const(strong seed-42 value actually in this task's
leaderboard row), scale=(strong-weak)/ln(9) -- so the strong baseline scores EXACTLY
0.5000 and the weak baseline scores ~0.1000.
"""
from mlsbench.scoring.dsl import *

term("fg_psnr_mild",
    col("fg_psnr_mild").higher().id()
    .sigmoid(ref=const(18.1959), scale=3.191571800321681))

term("fg_psnr_medium",
    col("fg_psnr_medium").higher().id()
    .sigmoid(ref=const(19.6505), scale=3.9750602146407306))

term("fg_psnr_strong",
    col("fg_psnr_strong").higher().id()
    .sigmoid(ref=const(18.2842), scale=3.343308679400373))

setting("mild", weighted_mean(("fg_psnr_mild", 1.0)))
setting("medium", weighted_mean(("fg_psnr_medium", 1.0)))
setting("strong", weighted_mean(("fg_psnr_strong", 1.0)))

task(gmean("mild", "medium", "strong"))
