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
"""
from mlsbench.scoring.dsl import *

# per-setting logistic (midpoint = geomean(weak, strong) SAD; lower is better;
# scale = (weak_avg - strong_avg) / ln(9))
_CAL = {
    "medium": (0.347614, 0.076483),
    "wide": (0.707536, 0.071431),
    "xwide": (1.284045, 0.177997)
}

for _s, (_mid, _scale) in _CAL.items():
    term(f"sad_{_s}",
        col(f"sad_{_s}").lower().id()
        .sigmoid(ref=const(_mid), scale=_scale))
    setting(_s, weighted_mean((f"sad_{_s}", 1.0)))

task(gmean("medium", "wide", "xwide"))
