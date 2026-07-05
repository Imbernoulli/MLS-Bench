"""Score spec for cv-matting-norm.

Alpha SAD (LOWER is better; /1000, measured in the trimap UNKNOWN band) of a matting
net trained a short fine-tune on REAL PPM-100 composites with the agent's no normalisation (identity) vs BatchNorm (cross-image statistics).
The default is a no normalisation (identity); the strong reference is a BatchNorm (cross-image statistics). The score is the
gmean over THREE trimap-width settings (medium=band6 / wide=band9 / xwide=band12),
each with a logistic midpoint = geometric mean of the reproduced weak & strong SAD.

MEASURED anchors (B0 8xH200, torch 2.4.1, REAL PPM-100 alpha matting, CROSS-SEED
42/123, seed-averaged, full budget 400 iters) — reproduced baselines, alpha SAD in
the trimap unknown band:
                  medium(6)   wide(9)   xwide(12)
  identity       0.4159  0.9442  1.7256  (weak default)
  batch          0.2712  0.6241  1.0539  (strong = SOTA)
weak<strong holds ROBUSTLY on ALL THREE settings, BOTH seeds, with the LARGEST
margins of any matting surface (seed42: medium 0.4169>0.2561, wide 0.9523>0.6578,
xwide 1.7665>1.1214; seed123: medium 0.4150>0.2862, wide 0.9360>0.5904, xwide
1.6847>0.9864). Clean cross-seed re-anchor, ordering unchanged from the earlier
synthetic-data pass.
"""
from mlsbench.scoring.dsl import *

# per-setting logistic (midpoint = geomean(weak, strong) SAD; lower is better;
# scale = (weak_avg - strong_avg) / ln(9))
_CAL = {
    "medium": (0.335835, 0.065901),
    "wide": (0.767622, 0.145661),
    "xwide": (1.348558, 0.305704)
}

for _s, (_mid, _scale) in _CAL.items():
    term(f"sad_{_s}",
        col(f"sad_{_s}").lower().id()
        .sigmoid(ref=const(_mid), scale=_scale))
    setting(_s, weighted_mean((f"sad_{_s}", 1.0)))

task(gmean("medium", "wide", "xwide"))
