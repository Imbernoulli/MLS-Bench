"""Pending calibration for the pinned full-resolution inpainting protocol.

There are deliberately no baseline rows yet. With no trusted floor anchor, the
base evaluator uses the submitted value as its floor, so every otherwise valid
metric maps to exactly zero. Fresh complete worker-side anchors may replace this
pending calibration; failed or incomplete verification remains zero.
"""
from mlsbench.scoring.dsl import *

for _setting in ("small", "large", "strokes"):
    _metric = f"hole_l1_{_setting}"
    _lower = f"{_metric}_domain_lower"
    _upper = f"{_metric}_domain_upper"
    term(
        _metric,
        # -1 is outside the legal L1 domain. It avoids the base evaluator's
        # bound==floor special case at a perfect raw value of zero.
        col(_metric).lower().id().bounded_power(bound=const(-1.0)),
    )
    term(_lower, penalty_lower(col(_metric), target=0.0, sharpness=1.0))
    term(_upper, penalty_upper(col(_metric), target=1.0, sharpness=1.0))
    setting(
        _setting,
        weighted_mean((_metric, 1.0)),
        constraints=[_lower, _upper],
    )

task(gmean("small", "large", "strokes"))
