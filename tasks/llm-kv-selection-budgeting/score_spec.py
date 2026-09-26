"""Score spec for llm-kv-selection-budgeting.

Each workload reports benchmark accuracy, the retained KV fraction and the
runtime. The score combines accuracy and cache reduction with weights 3:1
under the fixed retained-fraction budget; full-cache anchors are kept
visible but fail the retained-fraction constraint.

Anchor notes. Every term's floor is the worst baseline for its direction
(``BaselineAnchors.worst_for``): the lowest accuracy and the largest retained
fraction (``full_attention``, 1.0).

- runtime is reported but not scored. The baseline runtimes differ by only
  ~5-7 % per workload, which is within run-to-run variance, while wall-clock
  time moves by ~2x across hardware (an H100 sandbox runs the reference
  methods in about half the leaderboard runtime), so a runtime term would
  score the machine rather than the method.
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
- quality floor: reduction is only worth something while the kept entries
  still carry the context. Each workload's score is multiplied by a soft
  lower-bound penalty on accuracy at half of the weakest sparse reference
  method's leaderboard accuracy, decaying to 0.01 at zero accuracy. Every
  sparse baseline sits at twice the target or more, so the penalty is 1.0
  for them. Keeping no prefill entries at all (measured with the 3B model:
  hotpotqa 5.1, passage retrieval 2.6, repobench 18.0, longbench-v2 3.8,
  gsm8k 0.7 on 300 problems) otherwise collected the full reduction term
  and scored 0.20 against 0.27 for the best baseline; it now scores 0.03.
"""
import math

from mlsbench.scoring.dsl import *


# Accuracy gain (points) over the worst baseline that scores 0.5 on
# longbench-v2: 20 of 503 questions, ~2 binomial standard errors.
LONGBENCH_V2_HALF_SCORE_GAIN = 4.0

# Leaderboard accuracy of the weakest sparse (compressed-cache) baseline per
# workload; the quality-floor target is half of it.
WEAKEST_SPARSE_BASELINE_ACCURACY = {
    "longbench-hotpotqa": 25.586376,  # streamingllm
    "longbench-passage-retrieval": 50.288889,  # expected_attention
    "longbench-repobench": 40.87,  # lagkv
    "longbench-v2": 29.025845,  # lagkv
    "gsm8k": 1.743745,  # streamingllm
}


def add_setting(name: str, quality_half_score_gain=None) -> None:
    slug = name.replace("-", "_")
    quality_metric = f"final_score_{name}"
    retained_metric = f"mean_retained_fraction_{name}"
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
    # 1.0 at or above the target, exp(-ln(100)) = 0.01 at zero accuracy.
    floor_target = 0.5 * WEAKEST_SPARSE_BASELINE_ACCURACY[name]
    term(
        f"quality_floor_{slug}",
        penalty_lower(col(quality_metric).id(), target=floor_target, sharpness=math.log(100.0) / floor_target),
    )
    setting(
        name,
        weighted_mean(
            (f"quality_{slug}", 6.0),
            (f"reduction_{slug}", 2.0),
        ),
        constraints=[f"budget_{slug}", f"quality_floor_{slug}"],
    )


add_setting("longbench-hotpotqa")
add_setting("longbench-passage-retrieval")
add_setting("longbench-repobench")
add_setting("longbench-v2", quality_half_score_gain=LONGBENCH_V2_HALF_SCORE_GAIN)
add_setting("gsm8k")

task(gmean("longbench-hotpotqa", "longbench-passage-retrieval", "longbench-repobench", "longbench-v2", "gsm8k"))
