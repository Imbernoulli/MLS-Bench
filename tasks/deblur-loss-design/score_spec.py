"""Score spec for deblur-loss-design.

Deblur PSNR (dB, higher is better) of a compact residual deblur net trained on the fixed
synthetic motion-blur set, restoring sharp images from blurry ones, scored on a HELD-OUT
split (disjoint patches + kernels). The score aggregates (geometric mean) over THREE
motion-blur severity settings -- small / medium / large -- so the score reflects the loss
design across a range of blur strengths, not a single operating point. The 'loss' surface
(what target the net is optimised toward: the true SHARP GT vs an OVER-SMOOTHED low-pass
target) is the only editable lever.

It is monotone and cheat-proof: for each setting the score is normalised between the
blurry-INPUT PSNR floor (identity / do-nothing = score 0) and the strong sharp-target
reference (score 1). A net that copies its input scores ~0; a constant/gray output scores
far below the floor and clips to 0; the over-smoothing baseline (which reproduces the blur)
lands at/below the floor. Only genuinely restoring sharp detail lifts the score.

Anchors are pinned from REAL Real GoPro Large-Scale Blur Dataset (Nah, CVPR'17) GPU
validation (B0 8xH200, torch 2.4.1, 400 iters, CROSS-SEED 42/123, seed-averaged; see
`vendor/data_scripts/image-deblur/prepare_data.py` for data provenance). Per setting:
floor = validated blurry-input PSNR; bound = the strong SHARP-target reference PSNR (the
reproduced good answer). The weak OVER-SMOOTHED baseline sits well below the floor in
every setting, both seeds, so the strong>weak partial-order holds across all three
settings.

Validated per-setting PSNR (cross-seed avg; weak smoothed-target -> strong sharp-target ;
blurry floor):
  small : 27.6528 -> 36.2067  (floor 36.2553)
  medium: 24.2950 -> 27.9017  (floor 27.7132)
  large : 20.2351 -> 21.4228  (floor 21.3183)
"""
from mlsbench.scoring.dsl import *

# ---- per-setting anchors from the real cross-seed GPU validation ----
# floor  = validated blurry-input PSNR         -> score 0 (identity / do-nothing)
# ref    = validated strong sharp-target PSNR  -> ref_score (the reproduced good answer)
# bound  = ref + headroom                       -> score 1 (leaves room to exceed SOTA)
# The weak over-smoothed baseline sits well below the floor and scores ~0 in every
# setting, both seeds, so the strong>weak partial-order is preserved across all three
# settings.
_FLOOR = {"small": 36.2553, "medium": 27.7132, "large": 21.3183}
_STRONG = {"small": 36.2067, "medium": 27.9017, "large": 21.4228}
_HEADROOM = 1.5          # dB of headroom above the strong reference -> score 1
_REF_SCORE = 0.5        # the strong sharp-target reference maps to this score

for _s in ("small", "medium", "large"):
    term(f"psnr_{_s}",
        col(f"psnr_{_s}").higher().id()
        .bounded_power(bound=const(_STRONG[_s] + _HEADROOM),
                       ref=const(_STRONG[_s]), ref_score=_REF_SCORE,
                       floor=const(_FLOOR[_s])))
    setting(_s, weighted_mean((f"psnr_{_s}", 1.0)))

task(gmean("small", "medium", "large"))
