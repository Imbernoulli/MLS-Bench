"""Score spec for cv-matting-arch.

Alpha SAD (LOWER is better; /1000, measured in the trimap UNKNOWN band) of a matting
net trained a short fine-tune on REAL PPM-100 composites with the agent's PLAIN encoder-decoder vs DIM deep-matting (skips + refinement, Xu 2017 = SOTA).
The default is a PLAIN encoder-decoder; the strong reference is a DIM deep-matting (skips + refinement, Xu 2017 = SOTA). The score is the
gmean over THREE trimap-width settings (medium=band6 / wide=band9 / xwide=band12),
each with a logistic midpoint = geometric mean of the reproduced weak & strong SAD.

MEASURED anchors (B0 8xH200, torch 2.4.1, REAL PPM-100 alpha matting, CROSS-SEED
42/123, seed-averaged, full budget 400 iters) — reproduced baselines, alpha SAD in
the trimap unknown band:
                  medium(6)   wide(9)   xwide(12)
  constant       3.2950  4.3704  5.3597  (degenerate; crushed on all 3, seed-invariant)
  plain          0.4288  0.7531  1.3797  (weak default)
  dim            0.2613  0.6364  1.0736  (strong = SOTA)
weak<strong (plain worse than dim) holds ROBUSTLY on ALL THREE settings, BOTH
seeds (seed42: medium 0.4257>0.2666, wide 0.7314>0.6458, xwide 1.2636>1.0482;
seed123: medium 0.4318>0.2560, wide 0.7748>0.6270, xwide 1.4958>1.0990). Clean
cross-seed re-anchor, ordering unchanged from the earlier synthetic-data pass.
"""
from mlsbench.scoring.dsl import *

# per-setting logistic (midpoint = geomean(weak, strong) SAD; lower is better;
# scale = (weak_avg - strong_avg) / ln(9), matching the mono3d convention but
# with ref = geomean(weak, strong) rather than ref = strong)
_CAL = {
    "medium": (0.334712, 0.076210),
    "wide": (0.692295, 0.053112),
    "xwide": (1.217064, 0.139312)
}

for _s, (_mid, _scale) in _CAL.items():
    term(f"sad_{_s}",
        col(f"sad_{_s}").lower().id()
        .sigmoid(ref=const(_mid), scale=_scale))
    setting(_s, weighted_mean((f"sad_{_s}", 1.0)))

task(gmean("medium", "wide", "xwide"))
