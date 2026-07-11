"""Score spec for inr-fourier-frequency.

Full-grid reconstruction PSNR in dB (higher is better) after the fixed
real-data protocol: three pinned Kodak 256x256 crops, 65,536 coordinates
per setting, 2,000 full-batch Adam steps, seed 0, and one GPU per cell.
The editable surface is a finite literal Fourier-feature sigma and is never
executed as agent Python. Every fresh endpoint completed with rc=0 and unique
data, final-step, metric, and completion proof.

Fresh measured endpoints (8-card H200 allocation, 8 independent single-GPU
cells plus one second-wave cell, 2026-07-10):
    low: sigma_low=31.423719, sigma_high=31.916663, sigma_tuned=38.989245
    medium: sigma_low=22.561327, sigma_high=26.922403, sigma_tuned=39.083427
    high: sigma_low=20.417760, sigma_high=30.099870, sigma_tuned=33.648325
"""
from mlsbench.scoring.dsl import *

term("psnr_low", col("psnr_low").higher().id().sigmoid(ref=const(38.989245), scale=3.44321926763261))
term("psnr_medium", col("psnr_medium").higher().id().sigmoid(ref=const(39.083427), scale=7.51953176312564))
term("psnr_high", col("psnr_high").higher().id().sigmoid(ref=const(33.648325), scale=6.02148962671805))

setting("low", weighted_mean(("psnr_low", 1.0)))
setting("medium", weighted_mean(("psnr_medium", 1.0)))
setting("high", weighted_mean(("psnr_high", 1.0)))

task(gmean("low", "medium", "high"))
