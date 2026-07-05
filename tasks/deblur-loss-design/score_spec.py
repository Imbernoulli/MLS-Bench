"""Score spec for deblur-loss-design.

Deblur PSNR (dB, higher is better) of a compact residual deblur net trained on the fixed
synthetic motion-blur set, restoring sharp images from blurry ones, scored on a HELD-OUT
split (disjoint patches + kernels). The score aggregates (geometric mean) over THREE
motion-blur severity settings -- small / medium / large -- so the score reflects the loss
design across a range of blur strengths, not a single operating point. The 'loss' surface
(what target the net is optimised toward: the true SHARP GT vs an OVER-SMOOTHED low-pass
target) is the only editable lever.

The default optimises toward the true SHARP GT (strong / SOTA); the over-smoothed low-pass
target is the weak baseline (it lands far below the blurry-input floor because it actively
removes detail rather than restoring it).

Anchors are pinned from REAL Real GoPro Large-Scale Blur Dataset (Nah, CVPR'17) GPU
validation (B0 8xH200, torch 2.4.1, 400 iters, CROSS-SEED 42/123; see
`vendor/data_scripts/image-deblur/prepare_data.py` for data provenance). The strong
sharp-target baseline robustly beats the weak over-smoothed baseline on all three settings,
both seeds:
  seed 42  : small  sharp 36.1730 > smoothed 27.7393
             medium sharp 27.9089 > smoothed 24.3202
             large  sharp 21.4275 > smoothed 20.2663
  seed 123 : small  sharp 36.2403 > smoothed 27.5662
             medium sharp 27.8944 > smoothed 24.2698
             large  sharp 21.4180 > smoothed 20.2039

NOTE: the previous version of this spec used `bounded_power` with `floor` = the
blurry-INPUT PSNR. That floor (e.g. 36.2553 for `small`) sits ABOVE the strong sharp-target
PSNR at that operating point (the net barely improves on a nearly-sharp input at low blur
severity), so `ref <= floor` and the term degenerated (r_ref~0, gamma fallback, sharp
scoring far below 0.5). Per the project's SOTA=0.5 discipline (ref = strong-baseline
SEED-42 value, `scale=(strong-weak)/ln(9)`, matching stereo-disparity-range), this is now a
plain sigmoid keyed off the two REAL baselines (not the blurry-input floor), so `sharp`
scores exactly 0.5 at seed 42 and `smoothed` lands ~0.1.
"""
from mlsbench.scoring.dsl import *

# per-setting logistic: ref = strong (sharp) PSNR at seed 42 -> score 0.5;
# scale = (strong_seed42 - weak_seed42) / ln(9) so weak (smoothed) at seed 42 -> ~0.1
_CAL = {
    "small": (36.1730, 3.8383422828013796),
    "medium": (27.9089, 1.6332877562978652),
    "large": (21.4275, 0.5284848949795405),
}

for _s, (_ref, _scale) in _CAL.items():
    term(f"psnr_{_s}",
        col(f"psnr_{_s}").higher().id()
        .sigmoid(ref=const(_ref), scale=_scale))
    setting(_s, weighted_mean((f"psnr_{_s}", 1.0)))

task(gmean("small", "medium", "large"))
