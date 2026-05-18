"""Score spec for llm-kv-adaptive-quantization."""
from mlsbench.scoring.dsl import *


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
    setting(
        name,
        weighted_mean(
            (f"final_score_{slug}", 6.0),
            (f"kv_compression_ratio_{slug}", 4.0),
        ),
    )


add_setting("longbench-hotpotqa")
add_setting("longbench-passage-retrieval")
add_setting("longbench-repobench")
add_setting("needlebench-niah")
add_setting("gsm8k")

task(gmean("longbench-hotpotqa", "longbench-passage-retrieval", "longbench-repobench", "needlebench-niah", "gsm8k"))
