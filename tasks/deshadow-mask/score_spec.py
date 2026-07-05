"""Score spec for deshadow-mask.

SHADOW-REGION PSNR (dB, higher is better) of a FIXED residual deshadower (composite L1+SSIM
loss up-weighted in the shadow region) trained on the REAL ISTD shadow/shadow-free/mask photo
triplets (Wang, Li & Yang, CVPR 2018; HF Hub `Donghyun99/ISTD` mirror), removing a cast shadow
to match the clean GT, scored on a HELD-OUT split (ISTD's own disjoint-scene train/val) over
THREE cast-shadow severities (light / medium / heavy -- terciles of measured shadow attenuation
ratio mean(shadow_img[mask])/mean(shadow_free_target[mask]), pooled across ISTD train+test).
The metric is measured ONLY inside the shadow region, so a method that merely copies the LIT
region cannot win. The ONLY editable lever is whether the soft shadow MASK is fed to the net as
a 4th input channel: weak (use_mask=False, blind DeshadowNet-style) -> strong (use_mask=True,
mask-guided SP+M-Net recovery). It is monotone and cheat-proof: the harness also reports the
shadowed-INPUT copy floor, so a net that copies its input scores psnr==shadow_psnr and a
constant / all-white / all-black output scores far below it.

Fresh GPU re-anchor on REAL ISTD data (k1 H20, torch 2.4.0, package-default 400 iters,
cross-seed 42/123, vendor/image-deshadow/run_anchors.py -> anchor_real_full.tsv). Strong
(mask-guided) beats weak (blind) on EVERY setting and BOTH seeds individually, by a wide,
non-trivial margin -- clean, no diagnostic needed:
  light  seed42  weak=26.2982 strong=32.4008  seed123 weak=27.0981 strong=34.6657
  medium seed42  weak=23.6229 strong=29.7331  seed123 weak=25.3845 strong=32.1238
  heavy  seed42  weak=22.7424 strong=27.8355  seed123 weak=23.5114 strong=28.4755

Anchors below use the SEED-42 values (matching the seed-42 leaderboard.csv rows that get
scored), so `baseline:mask` (strong) scores exactly ref_score=0.5 and `baseline:nomask` (weak)
floors near 0 -- the seed-123 numbers above are cross-seed provenance confirming the ordering
is robust, not the pinned anchor. floor = weak (score 0); ref = strong (score 0.5); bound =
ref + headroom (score 1).
"""
from mlsbench.scoring.dsl import *

# per-setting weak (no mask) and strong (mask-guided) PSNR, SEED-42 (matches leaderboard rows).
_WEAK = {"light": 26.2982, "medium": 23.6229, "heavy": 22.7424}
_STRONG = {"light": 32.4008, "medium": 29.7331, "heavy": 27.8355}
# floor (score 0) = the weak (blind) reference; ref = the strong mask-guided reference.
_FLOOR = {_s: _WEAK[_s] for _s in _WEAK}
_HEADROOM = 1.0          # dB above the strong reference -> score 1
_REF_SCORE = 0.5        # the strong mask-guided reference maps to this score

for _s in ("light", "medium", "heavy"):
    term(f"psnr_{_s}",
        col(f"psnr_{_s}").higher().id()
        .bounded_power(bound=const(_STRONG[_s] + _HEADROOM),
                       ref=const(_STRONG[_s]), ref_score=_REF_SCORE,
                       floor=const(_FLOOR[_s])))
    setting(_s, weighted_mean((f"psnr_{_s}", 1.0)))

task(gmean("light", "medium", "heavy"))
