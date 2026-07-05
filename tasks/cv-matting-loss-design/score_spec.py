"""Score spec for cv-matting-loss-design.

Alpha SAD (LOWER is better; /1000, measured in the trimap UNKNOWN band) of a matting
net trained a short fine-tune on REAL PPM-100 composites with the agent's whole-image alpha-L1 vs unknown-band L1 + composition (Deep Image Matting, Xu 2017).
The default is a whole-image alpha-L1; the strong reference is a unknown-band L1 + composition (Deep Image Matting, Xu 2017). The score is the
gmean over THREE trimap-width settings (medium=band6 / wide=band9 / xwide=band12),
each with a logistic midpoint = geometric mean of the reproduced weak & strong SAD.

MEASURED anchors (B0 8xH200, torch 2.4.1, REAL PPM-100 alpha matting, CROSS-SEED
42/123, seed-averaged, full budget 400 iters) — reproduced baselines, alpha SAD in
the trimap unknown band:
                  medium(6)   wide(9)   xwide(12)
  whole_l1       0.3421  0.7642  1.2797  (weak default)
  unk_comp       0.2683  0.6355  1.0745  (strong = SOTA)
weak<strong holds ROBUSTLY on ALL THREE settings, BOTH seeds (seed42: medium
0.3434>0.2528, wide 0.7973>0.6584, xwide 1.3005>1.1032; seed123: medium
0.3408>0.2839, wide 0.7311>0.6126, xwide 1.2590>1.0457). Clean cross-seed
re-anchor, ordering unchanged from the earlier synthetic-data pass.

SOTA=0.5 anchor convention (matches stereo-disparity-range / mono3d-depth-cue):
"0.5 is the strongest baseline" -- ref = the STRONG (unk_comp) baseline's
SEED-42 value (NOT geomean(weak,strong), NOT seed-averaged) so unk_comp scores
exactly 0.5 at seed 42; scale = (weak_seed42 - strong_seed42) / ln(9) so the
weak (whole_l1) baseline lands ~0.1 at seed 42.
"""
from mlsbench.scoring.dsl import *

# per-setting logistic: ref = strong (unk_comp) SAD at seed 42 -> score 0.5;
# scale = (weak_seed42 - strong_seed42) / ln(9) so weak (whole_l1) at seed 42 -> ~0.1
_CAL = {
    "medium": (0.2528, 0.041234),
    "wide": (0.6584, 0.063216),
    "xwide": (1.1032, 0.089795)
}

for _s, (_mid, _scale) in _CAL.items():
    term(f"sad_{_s}",
        col(f"sad_{_s}").lower().id()
        .sigmoid(ref=const(_mid), scale=_scale))
    setting(_s, weighted_mean((f"sad_{_s}", 1.0)))

task(gmean("medium", "wide", "xwide"))
