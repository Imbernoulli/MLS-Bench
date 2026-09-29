"""Score spec for llm-kv-adaptive-quantization."""
import math

from mlsbench.scoring.dsl import *

# Quality gate on the compression term. The quality term's floor is the worst
# baseline, so any record below it scores 0 on quality and was ranked by its
# compression alone: a zeroed cache with a 1-bit layout (quality 0, ratio 16)
# scored 0.40, above squat_subspace_4bit (0.39) and kivi_overlap_4bit (0.195).
# Each setting is therefore multiplied by exp(-sharpness * (target - quality))
# below a target of 90% of the worst baseline's quality on that workload
# (leaderboard: hotpotqa 35.78, passage retrieval 56.63, repobench 43.74,
# NIAH 53.74, GSM8K 31.77), with the sharpness set so that quality 0 gives
# x0.01. Every baseline sits above its target (penalty 1.0); quality at 80% of
# the worst baseline gives x0.60, at 50% x0.13.
QUALITY_GATE_TARGETS = {
    "longbench-hotpotqa": 32.20,
    "longbench-passage-retrieval": 50.97,
    "longbench-repobench": 39.37,
    "needlebench-niah": 48.37,
    "gsm8k": 28.59,
}
QUALITY_GATE_AT_ZERO = 0.01


def add_setting(name: str, ref: float = 50.0) -> None:
    slug = name.replace("-", "_")
    quality_col = f"final_score_{name}"
    compression_col = f"kv_compression_ratio_{name}"
    term(
        f"final_score_{slug}",
        col(quality_col).higher().id().bounded_power(
            bound=100.0,
            ref=bl_best(quality_col),
            ref_score=0.5,
        ),
    )
    term(
        f"kv_compression_ratio_{slug}",
        col(compression_col).higher().id().bounded_power(
            bound=8.0,
            ref=4.0,
            ref_score=0.5,
        ),
    )
    target = QUALITY_GATE_TARGETS[name]
    term(
        f"quality_gate_{slug}",
        penalty_lower(
            col(quality_col).higher().id(),
            target=target,
            sharpness=-math.log(QUALITY_GATE_AT_ZERO) / target,
        ),
    )
    setting(
        name,
        weighted_mean(
            (f"final_score_{slug}", 6.0),
            (f"kv_compression_ratio_{slug}", 4.0),
        ),
        constraints=[f"quality_gate_{slug}"],
    )


add_setting("longbench-hotpotqa")
add_setting("longbench-passage-retrieval")
add_setting("longbench-repobench")
add_setting("needlebench-niah")
add_setting("gsm8k")

task(gmean("longbench-hotpotqa", "longbench-passage-retrieval", "longbench-repobench", "needlebench-niah", "gsm8k"))
