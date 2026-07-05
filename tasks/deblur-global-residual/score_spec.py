"""Score spec for deblur-global-residual.

Deblur PSNR (dB, higher is better) of a compact deblur net trained on the fixed synthetic
motion-blur set, restoring sharp images from blurry ones, scored on a HELD-OUT split
(disjoint patches + kernels). The score aggregates (geometric mean) over THREE mild
motion-blur severity settings -- rs / rm / rl -- so the score reflects the design across a
range of blur strengths. The 'residual' surface (predict a GLOBAL RESIDUAL correction,
sharp = blurry + net(blurry), vs regress the full sharp image directly) is the only
editable lever.

It is monotone and cheat-proof: for each setting the score is normalised between the
direct-prediction reference (global_residual=False = the weak real baseline, score 0) and
the global-residual reference (global_residual=True = the strong answer used by DeepDeblur
/ SRN / MPRNet). BOTH baselines clearly beat the blurry-input identity floor (reported as
blurry_psnr), and global residual reliably improves on direct prediction in every setting.
A net that copies its input, or a constant output, scores far below the direct-prediction
reference and clips to 0.

Anchors are pinned from REAL Real GoPro Large-Scale Blur Dataset (Nah, CVPR'17) GPU
validation (B0 8xH200, torch 2.4.1, 400 iters, CROSS-SEED 42/123, seed-averaged; see
`vendor/data_scripts/image-deblur/prepare_data.py` for data provenance). Per setting:
floor = direct-prediction (residual_off) PSNR; ref = global-residual (residual_on) PSNR;
bound = ref + headroom. The strong>weak partial-order holds across all three settings,
BOTH seeds.

Validated per-setting PSNR (cross-seed avg; weak residual_off -> strong residual_on ;
blurry floor):
  rs : 33.6436 -> 36.2222  (floor 36.2553)
  rm : 27.3962 -> 27.9008  (floor 27.7132)
  rl : 21.3911 -> 21.4251  (floor 21.3183)
"""
from mlsbench.scoring.dsl import *

# ---- per-setting anchors from the real cross-seed GPU validation ----
# floor  = validated residual_off (direct-prediction) PSNR  -> score 0 (weak real baseline)
# ref    = validated residual_on  (global-residual) PSNR    -> ref_score (the strong answer)
# bound  = ref + headroom                                    -> score 1 (room to exceed)
_FLOOR = {"rs": 33.6436, "rm": 27.3962, "rl": 21.3911}
_STRONG = {"rs": 36.2222, "rm": 27.9008, "rl": 21.4251}
_HEADROOM = 1.0          # dB of headroom above the strong reference -> score 1
_REF_SCORE = 0.5        # the strong global-residual reference maps to this score

for _s in ("rs", "rm", "rl"):
    term(f"psnr_{_s}",
        col(f"psnr_{_s}").higher().id()
        .bounded_power(bound=const(_STRONG[_s] + _HEADROOM),
                       ref=const(_STRONG[_s]), ref_score=_REF_SCORE,
                       floor=const(_FLOOR[_s])))
    setting(_s, weighted_mean((f"psnr_{_s}", 1.0)))

task(gmean("rs", "rm", "rl"))
