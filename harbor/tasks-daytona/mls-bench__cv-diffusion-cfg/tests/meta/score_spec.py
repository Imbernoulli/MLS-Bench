"""Score spec for cv-diffusion-cfg.

Three settings (SD models): sd15, sd20, sdxl. Scoring uses FID only
(lower is better, bound=0). Refs are the lowest baseline FID per setting.

Baseline FIDs (single seed=42):
  cfg:      24.2 / 25.0 / 25.9
  cfgpp:    23.9 / 24.6 / 26.0
  zeroinit: 23.5 / 24.1 / 25.9
"""
from mlsbench.scoring.dsl import *

term("fid_sd15",
    col("fid_sd15").lower().id()
    .bounded_power(bound=0.0))

term("fid_sd20",
    col("fid_sd20").lower().id()
    .bounded_power(bound=0.0))

term("fid_sdxl",
    col("fid_sdxl").lower().id()
    .bounded_power(bound=0.0))

setting("sd15", weighted_mean(("fid_sd15", 1.0)))
setting("sd20", weighted_mean(("fid_sd20", 1.0)))
setting("sdxl", weighted_mean(("fid_sdxl", 1.0)))

task(gmean("sd15", "sd20", "sdxl"))
