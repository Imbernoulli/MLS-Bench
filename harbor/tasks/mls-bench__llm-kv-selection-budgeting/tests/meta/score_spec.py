"""Score spec for llm-kv-selection-budgeting.

Each workload reports benchmark accuracy, runtime, and the retained KV fraction.
The score combines accuracy, runtime, and cache reduction with weights 6:2:2
under the fixed retained-fraction budget; full-cache anchors are
kept visible but fail the retained-fraction constraint.
"""
from mlsbench.scoring.dsl import *


def add_setting(name: str) -> None:
    slug = name.replace("-", "_")
    quality_metric = f"final_score_{name}"
    retained_metric = f"mean_retained_fraction_{name}"
    runtime_metric = f"runtime_seconds_{name}"
    term(
        f"quality_{slug}",
        col(quality_metric).higher().id().bounded_power(
            bound=100.0,
            ref=bl_best(quality_metric),
            ref_score=0.5,
        ),
    )
    term(
        f"time_{slug}",
        # Baseline anchors are stored as raw min/max; for lower-is-better
        # metrics, bl_worst(raw min) is the best baseline calibration point.
        col(runtime_metric).lower().id().sigmoid(
            ref=bl_worst(runtime_metric),
            ref_score=0.5,
        ),
    )
    term(
        f"reduction_{slug}",
        col(retained_metric).lower().id().bounded_power(
            bound=0.0,
            ref=bl_worst(retained_metric),
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
add_setting("longbench-v2")
add_setting("gsm8k")

task(gmean("longbench-hotpotqa", "longbench-passage-retrieval", "longbench-repobench", "longbench-v2", "gsm8k"))
