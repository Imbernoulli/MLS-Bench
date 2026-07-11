"""Machine-translation score over three complete OPUS-100 test splits.

Each direction contributes corpus sacreBLEU (higher is better, theoretical ceiling
100). Calibration is derived only from fresh baselines evaluated on the same pinned
full splits; old sliced-split anchors are intentionally invalid.
"""
from mlsbench.scoring.dsl import *


for _direction in ("de_en", "fr_en", "ru_en"):
    term(
        f"bleu_{_direction}",
        col(f"bleu_{_direction}").higher().id().bounded_power(bound=100.0),
    )
    setting(_direction, weighted_mean((f"bleu_{_direction}", 1.0)))

task(gmean("de_en", "fr_en", "ru_en"))
