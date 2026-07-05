"""Score spec for deshadow-mask-guidance.

SHADOW-REGION PSNR (dB, higher is better) of a residual deshadower (a FIXED formulation,
composite L1+SSIM loss up-weighted in the shadow region) trained on the REAL ISTD shadow/
shadow-free/mask photo triplets (Wang, Li & Yang, CVPR 2018; HF Hub `Donghyun99/ISTD` mirror),
removing a cast shadow to match the clean GT, scored on a HELD-OUT split (ISTD's own
disjoint-scene train/val) over THREE cast-shadow severities (light / medium / heavy --
terciles of measured shadow attenuation ratio). The metric is measured ONLY inside the shadow
region, so a method that merely copies the LIT region cannot win. The 'network' surface (BLIND
U-Net that sees only the shadowed RGB vs the MASK-GUIDED U-Net that also takes the shadow mask
as input -- the SP+M-Net physically-parameterised recovery) is the only editable lever. It is
monotone and cheat-proof: the harness also reports the shadowed-INPUT copy floor, so a net that
copies its input scores psnr==shadow_psnr and a constant / all-white / all-black output scores
far below it.

Fresh GPU re-anchor on REAL ISTD data (k1 H20, torch 2.4.0, package-default 400 iters,
cross-seed 42/123, vendor/image-deshadow/run_anchors.py -> anchor_real_full.tsv). unet_mask
(strong) beats unet_nomask (weak) on EVERY setting and BOTH seeds individually, by a wide,
non-trivial margin -- clean, no diagnostic needed:
  light  seed42  weak=26.8237 strong=33.0063  seed123 weak=27.3903 strong=34.5763
  medium seed42  weak=24.5654 strong=30.4833  seed123 weak=25.2578 strong=31.6453
  heavy  seed42  weak=22.6890 strong=27.2594  seed123 weak=24.0147 strong=28.9516

Anchors below use the SEED-42 values (matching the seed-42 leaderboard.csv rows that get
scored), so `baseline:network_unet_mask` (strong) scores exactly ref_score=0.5 and
`baseline:network_unet_nomask` (weak) scores ~0.1 -- the seed-123 numbers above are cross-seed
provenance confirming the ordering is robust, not the pinned anchor. Per-setting ref = strong
(unet_mask) seed-42 value; scale = (weak-strong)/ln(9) so weak scores ~0.1.

NOTE: leaderboard.csv model names must match config.json's baseline keys
(`network_unet_mask` / `network_unet_nomask`, not the bare `unet_mask`/`unet_nomask`) or
BaselineAnchors silently drops the rows and `mlsbench check` warns "no current baseline
anchor" -- fixed in this pass.
"""
from mlsbench.scoring.dsl import *

_STRONG = {"light": 33.0063, "medium": 30.4833, "heavy": 27.2594}

for _s, _scale in (("light", 2.813823), ("medium", 2.693352), ("heavy", 2.080079)):
    term(f"psnr_{_s}",
        col(f"psnr_{_s}").higher().id().sigmoid(ref=const(_STRONG[_s]), scale=_scale))
    setting(_s, weighted_mean((f"psnr_{_s}", 1.0)))

task(gmean("light", "medium", "heavy"))
