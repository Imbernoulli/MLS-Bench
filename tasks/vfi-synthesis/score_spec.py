"""Score spec for vfi-synthesis.

Interpolation PSNR (dB, higher is better) of the SYNTHESIZED middle frame (t=0.5) vs the
true middle frame, for a compact FIXED VFI net trained on REAL Vimeo-90K triplet-
interpolation tiles (see vendor/data_scripts/video-frame-interp/prepare_data.py: TOFlow /
Vimeo-90K, Xue et al. IJCV 2019 -- genuine decoded video frames, tiled 64x64, terciled by
MEASURED per-tile Farneback motion magnitude). The score aggregates (geometric mean) over
TWO inter-frame MOTION-MAGNITUDE settings -- medium / large -- so it reflects the
synthesis strategy across displacements where motion compensation and occlusion
reasoning matter. The 'synthesis' surface (blend vs flow_warp vs learned) is the only
editable lever.

It is monotone and cheat-proof: for each setting the score is normalised between the
flow_warp reference (the weak-real, motion-compensated baseline -> score 0) and the
learned reference (Super-SloMo flow+refinement -> the strong / SOTA answer). BOTH
clearly beat the naive linear-blend floor (reported as blend_psnr); the naive blend
sits BELOW the flow_warp floor and clips to 0. A net that copies a single input frame
or outputs a constant scores far below flow_warp. The blend < flow_warp < learned
partial-order holds, CROSS-SEED (42, 123), on both shipped settings, and the
learned>flow_warp margin WIDENS with motion (more disocclusion): ~0.2 dB at medium,
~1.6 dB at large.

DROPPED SETTING (honest, cross-seed-confirmed real-data finding): the original THIRD
setting, `small` inter-frame motion, is EXCLUDED from scoring. On real Vimeo-90K small-
motion tiles the blend floor alone is already 41.16 dB (near this compact net's
effective ceiling), so there is essentially no disocclusion for the learned refinement
head to resolve -- and on BOTH seeds (42, 123) `flow_warp` (41.69 / 41.77 dB) beats
`learned` (41.32 / 41.47 dB), inverting the intended order. This reproduces cross-seed
(not a fluke) and mirrors the same near-ceiling-saturation failure mode documented for
image-matting's excluded narrow (width=2) trimap band. Per project mandate (never
HP-sweep to force monotonicity), `small` is dropped rather than papered over; see
vendor/video-frame-interp/anchors/README.md for full cross-seed numbers and rationale.
`scripts/vfi_small.sh` has been moved to `dropped_scripts/` for provenance.

Anchors are pinned from the REAL GPU cross-seed validation (B0 8xH200, 800 iters,
seeds 42 + 123, averaged; see vendor/video-frame-interp/anchors/README.md). Per
setting: floor = flow_warp PSNR (score 0); ref = learned PSNR (the strong reference);
bound = ref + headroom (score 1).

Validated per-setting PSNR, seed-avg (blend / flow_warp -> learned):
  medium: 28.59 / 32.43 -> 32.64
  large : 22.16 / 23.70 -> 25.35
"""
from mlsbench.scoring.dsl import *

# ---- per-setting anchors from the real cross-seed GPU validation ----
# floor = validated flow_warp PSNR (seed-avg)  -> score 0 (weak-real motion-compensated baseline)
# ref   = validated learned PSNR (seed-avg)    -> ref_score (the strong / SOTA answer)
# bound = ref + headroom                        -> score 1 (room to exceed)
_FLOOR = {"medium": 32.4315, "large": 23.7037}
_STRONG = {"medium": 32.6357, "large": 25.3458}
_HEADROOM = 1.0          # dB of headroom above the strong reference -> score 1
_REF_SCORE = 0.5        # the strong learned reference maps to this score

for _s in ("medium", "large"):
    term(f"psnr_{_s}",
        col(f"psnr_{_s}").higher().id()
        .bounded_power(bound=const(_STRONG[_s] + _HEADROOM),
                       ref=const(_STRONG[_s]), ref_score=_REF_SCORE,
                       floor=const(_FLOOR[_s])))
    setting(_s, weighted_mean((f"psnr_{_s}", 1.0)))

task(gmean("medium", "large"))
