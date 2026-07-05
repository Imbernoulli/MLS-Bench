"""Score spec for deblur-global-residual.

Deblur PSNR (dB, higher is better) of a compact deblur net trained on the fixed synthetic
motion-blur set, restoring sharp images from blurry ones, scored on a HELD-OUT split
(disjoint patches + kernels). The score aggregates (geometric mean) over THREE mild
motion-blur severity settings -- rs / rm / rl -- so the score reflects the design across a
range of blur strengths. The 'residual' surface (predict a GLOBAL RESIDUAL correction,
sharp = blurry + net(blurry), vs regress the full sharp image directly) is the only
editable lever.

The strong global-residual (residual_on) baseline robustly beats the weak direct-prediction
(residual_off) baseline on all three settings, both seeds, though the `rl` margin is small
in absolute PSNR (~0.03-0.05 dB on real GoPro at this severity).

Anchors are pinned from REAL Real GoPro Large-Scale Blur Dataset (Nah, CVPR'17) GPU
validation (B0 8xH200, torch 2.4.1, 400 iters, CROSS-SEED 42/123; see
`vendor/data_scripts/image-deblur/prepare_data.py` for data provenance):
  seed 42  : rs  residual_on 36.2077 > residual_off 32.7374
             rm  residual_on 27.9072 > residual_off 27.2255
             rl  residual_on 21.4380 > residual_off 21.3883
  seed 123 : rs  residual_on 36.2367 > residual_off 34.5497
             rm  residual_on 27.8943 > residual_off 27.5668
             rl  residual_on 21.4122 > residual_off 21.3938

NOTE: the previous version of this spec used `bounded_power` with floor=weak, bound=strong+
headroom. For `rl` the strong-weak margin (~0.03-0.05 dB) is tiny relative to the 1.0 dB
headroom, making `r_ref` pathologically close to 0 (solve_gamma degenerately falls back to
gamma=1 and warns). Per the project's SOTA=0.5 discipline (ref = strong-baseline SEED-42
value, `scale=(strong-weak)/ln(9)`, matching stereo-disparity-range), this is now a plain
sigmoid keyed off the two REAL baselines directly (no bound/headroom needed), so
`residual_on` scores exactly 0.5 at seed 42 and `residual_off` lands ~0.1 -- with no
pathological-ratio warning.
"""
from mlsbench.scoring.dsl import *

# per-setting logistic: ref = strong (residual_on) PSNR at seed 42 -> score 0.5;
# scale = (strong_seed42 - weak_seed42) / ln(9) so weak (residual_off) at seed 42 -> ~0.1
_CAL = {
    "rs": (36.2077, 1.5794015940815576),
    "rm": (27.9072, 0.3102550403957572),
    "rl": (21.4380, 0.02261944478167593),
}

for _s, (_ref, _scale) in _CAL.items():
    term(f"psnr_{_s}",
        col(f"psnr_{_s}").higher().id()
        .sigmoid(ref=const(_ref), scale=_scale))
    setting(_s, weighted_mean((f"psnr_{_s}", 1.0)))

task(gmean("rs", "rm", "rl"))
