"""Score spec for cv-diffusion-cfg.

Three settings (SD models): sd15, sd20, sdxl. Each setting scores FID
(lower is better, bound=0; refs are the lowest baseline FID per setting),
times a CLIP prompt-alignment floor (below).

Baseline FID / CLIP (single seed=42), sd15 / sd20 / sdxl:
  cfg:      23.65 / 24.29 / 25.74    0.3139 / 0.3174 / 0.3183
  cfgpp:    23.99 / 24.89 / 25.88    0.3133 / 0.3168 / 0.3185
  zeroinit: 22.76 / 23.31 / 25.49    0.3131 / 0.3170 / 0.3176
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

# CLIP floor. FID alone can be bought by giving up prompt alignment: sampling
# at a lower guidance scale (w=3 instead of 7.5) improved FID over every
# baseline while its CLIP score fell by ~0.008. A setting whose CLIP score
# (clip_<model>: ViT-B/32 image-prompt similarity, mean over the generated
# images) is below 0.99 x the lowest baseline CLIP is multiplied by
# exp(-200 * (target - clip)): x0.5 at 0.0035 below the target, x0.14 at 0.01.
# Every baseline clears it (margin >= 0.0031); the per-image CLIP std is ~0.03,
# so the standard error of the 3000-image mean is ~0.0006.
_CLIP_MIN_BASELINE = {"sd15": 0.3131, "sd20": 0.3168, "sdxl": 0.3176}
for _m, _clip_min in _CLIP_MIN_BASELINE.items():
    term(f"clip_floor_{_m}",
        penalty_lower(col(f"clip_{_m}").higher().id(),
                      target=round(0.99 * _clip_min, 4), sharpness=200.0))

setting("sd15", weighted_mean(("fid_sd15", 1.0)), constraints=["clip_floor_sd15"])
setting("sd20", weighted_mean(("fid_sd20", 1.0)), constraints=["clip_floor_sd20"])
setting("sdxl", weighted_mean(("fid_sdxl", 1.0)), constraints=["clip_floor_sdxl"])

task(gmean("sd15", "sd20", "sdxl"))
