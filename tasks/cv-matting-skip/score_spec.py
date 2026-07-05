"""Score spec for cv-matting-skip.

Alpha SAD (LOWER is better; /1000, measured in the trimap UNKNOWN band) of a matting
net trained a short fine-tune on REAL PPM-100 composites with the agent's drop-skip vs full concat skip (U-Net).
The default is a drop-skip; the strong reference is a full concat skip (U-Net). The score is the
gmean over THREE trimap-width settings (medium=band6 / wide=band9 / xwide=band12),
each with a logistic midpoint = geometric mean of the reproduced weak & strong SAD.

MEASURED anchors (B0 8xH200, torch 2.4.1, REAL PPM-100 alpha matting, CROSS-SEED
42/123, seed-averaged, full budget 400 iters) — reproduced baselines, alpha SAD in
the trimap unknown band:
                  medium(6)   wide(9)   xwide(12)
  drop           0.4416  0.7903  1.4944  (weak default)
  concat         0.2736  0.6334  1.1033  (strong = SOTA)
weak<strong holds ROBUSTLY on ALL THREE settings, BOTH seeds (seed42: medium
0.4545>0.2669, wide 0.7919>0.6960, xwide 1.5297>1.2061; seed123: medium
0.4288>0.2803, wide 0.7888>0.5708, xwide 1.4591>1.0005). Cross-seed data
resolves the earlier single-seed "medium inversion within noise" concern in
favour of the intended concat>drop ordering: clean re-anchor.

SOTA=0.5 anchor convention (matches stereo-disparity-range / mono3d-depth-cue):
"0.5 is the strongest baseline" -- ref = the STRONG (concat) baseline's
SEED-42 value (NOT geomean(weak,strong), NOT seed-averaged) so concat scores
exactly 0.5 at seed 42; scale = (weak_seed42 - strong_seed42) / ln(9) so the
weak (drop) baseline lands ~0.1 at seed 42.
"""
from mlsbench.scoring.dsl import *

# per-setting logistic: ref = strong (concat) SAD at seed 42 -> score 0.5;
# scale = (weak_seed42 - strong_seed42) / ln(9) so weak (drop) at seed 42 -> ~0.1
_CAL = {
    "medium": (0.2669, 0.085380),
    "wide": (0.6960, 0.043646),
    "xwide": (1.2061, 0.147277)
}

for _s, (_mid, _scale) in _CAL.items():
    term(f"sad_{_s}",
        col(f"sad_{_s}").lower().id()
        .sigmoid(ref=const(_mid), scale=_scale))
    setting(_s, weighted_mean((f"sad_{_s}", 1.0)))

task(gmean("medium", "wide", "xwide"))
