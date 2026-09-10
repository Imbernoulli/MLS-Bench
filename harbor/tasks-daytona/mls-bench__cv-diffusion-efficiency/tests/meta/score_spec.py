"""Score spec for cv-diffusion-efficiency.

Three settings (SD models): sd15, sd20, sdxl. Scoring uses FID only
(lower is better, bound=0). Refs are baseline median FIDs.
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
