"""Repository-wide full-split OPUS-100 translation-quality score.

All three baseline rows were measured at seed 42 on one H20 against every row of
the pinned 2,000-pair official test split.  Per direction, native greedy maps to
0.1 and beam-5 with no-repeat-3 maps to 0.5 through a smooth sigmoid.  Missing
metrics and failed commands are handled by the fail-closed evaluator, never by a
fallback score.  Every mt-* sibling uses the same frozen checkpoints, splits,
metric, and shared calibration so a given translation quality has one score.
"""
from mlsbench.scoring.dsl import *


_CALIBRATION = {
    "de_en": (31.900600, 7.048548500570563),
    "fr_en": (31.890343, 4.527782504627279),
    "ru_en": (31.241402, 5.381301539205705),
}

for _direction, (_strong, _scale) in _CALIBRATION.items():
    term(
        f"bleu_{_direction}",
        col(f"bleu_{_direction}").higher().id().sigmoid(
            ref=const(_strong), scale=_scale
        ),
    )
    setting(_direction, weighted_mean((f"bleu_{_direction}", 1.0)))

task(gmean("de_en", "fr_en", "ru_en"))
