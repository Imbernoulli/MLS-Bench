"""Score spec for llm-kv-selection-budgeting.

Each workload reports benchmark accuracy, runtime, and the retained KV fraction.
The score combines accuracy, runtime, and cache reduction with weights 6:2:2
under the fixed retained-fraction budget; full-cache anchors are
kept visible but fail the retained-fraction constraint.

Anchor notes. Every term's floor is the worst baseline for its direction
(``BaselineAnchors.worst_for``): the slowest runtime and the largest retained
fraction (``full_attention``, 1.0). ``bl_worst`` therefore resolves to that
same floor, so it must not be used as a calibration ref here.

- time: the baseline runtimes differ by only ~5-7 % per workload, which is
  within run-to-run and hardware wall-clock variance, so the scale is not
  calibrated from that spread. It is fixed in log-runtime space so that a
  2x speedup over the slowest baseline scores 0.5; noise-level differences
  move the term by a few hundredths, real speedups are rewarded smoothly.
- reduction: linear in the removed cache fraction between the full-cache
  floor (retained 1.0 -> 0) and the bound (retained 0.0 -> 1), i.e. the
  term equals the achieved compression ratio (ref 0.5 -> 0.5 gives gamma 1).
- quality on longbench-v2: all baselines sit near the 25 % chance level of
  this 4-way multiple-choice set (29.03 and 29.62 = 146 vs 149 of the 503
  questions). That 3-question spread is far below the binomial standard
  error (~2 points at p ~0.29, n = 503), and calibrating the default
  ``bl_best`` ref against it (r_ref ~0.008) lands in the pathological
  sigmoid fallback with a ~0.5-point scale. The scale is therefore fixed in
  accuracy points instead: a gain of 4 points (20 questions, ~2 binomial
  standard errors) over the worst baseline scores 0.5. The other quality
  terms keep the ``bl_best`` calibration (r_ref 0.11-0.31).
"""
import math

from mlsbench.scoring.dsl import *


# Accuracy gain (points) over the worst baseline that scores 0.5 on
# longbench-v2: 20 of 503 questions, ~2 binomial standard errors.
LONGBENCH_V2_HALF_SCORE_GAIN = 4.0


def add_setting(name: str, quality_half_score_gain=None) -> None:
    slug = name.replace("-", "_")
    quality_metric = f"final_score_{name}"
    retained_metric = f"mean_retained_fraction_{name}"
    runtime_metric = f"runtime_seconds_{name}"
    if quality_half_score_gain is None:
        quality = col(quality_metric).higher().id().bounded_power(
            bound=100.0,
            ref=bl_best(quality_metric),
            ref_score=0.5,
        )
    else:
        # Floor = worst baseline; 2 * sigmoid(gain / scale) - 1 = 0.5.
        quality = col(quality_metric).higher().id().sigmoid(
            scale=quality_half_score_gain / math.log(3.0),
        )
    term(f"quality_{slug}", quality)
    term(
        f"time_{slug}",
        # Floor = slowest baseline; 2 * sigmoid(ln 2 / scale) - 1 = 0.5.
        col(runtime_metric).lower().log().sigmoid(
            scale=math.log(2.0) / math.log(3.0),
        ),
    )
    term(
        f"reduction_{slug}",
        col(retained_metric).lower().id().bounded_power(
            bound=0.0,
            ref=0.5,
            ref_score=0.5,
        ),
    )
    term(
        f"budget_{slug}",
        penalty_upper(col(retained_metric).id(), target=0.25, sharpness=8.0),
    )
    setting(
        name,
        weighted_mean(
            (f"quality_{slug}", 6.0),
            (f"time_{slug}", 2.0),
            (f"reduction_{slug}", 2.0),
        ),
        constraints=[f"budget_{slug}"],
    )


add_setting("longbench-hotpotqa")
add_setting("longbench-passage-retrieval")
add_setting("longbench-repobench")
add_setting("longbench-v2", quality_half_score_gain=LONGBENCH_V2_HALF_SCORE_GAIN)
add_setting("gsm8k")

task(gmean("longbench-hotpotqa", "longbench-passage-retrieval", "longbench-repobench", "longbench-v2", "gsm8k"))
