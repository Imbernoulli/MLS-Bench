"""Score spec for mlsys-fused-attention.

This is a systems kernel task, so task-internal efficiency metrics are scored:
latency, TFLOPs, and speedup versus SDPA. They are paired with correctness and
max-difference terms so a fast incorrect kernel cannot win.
"""
from mlsbench.scoring.dsl import *

term("tflops_hdim64_seq4k",
    col("tflops_hdim64_seq4k").higher().id()
    .sigmoid())

term("latency_ms_hdim64_seq4k",
    col("latency_ms_hdim64_seq4k").lower().id()
    .bounded_power(bound=0.0))

term("max_diff_hdim64_seq4k",
    col("max_diff_hdim64_seq4k").lower().id()
    .bounded_power(bound=0.0))

term("correct_hdim64_seq4k",
    penalty_lower(col("correct_hdim64_seq4k").higher().id(),
                  target=1.0, sharpness=float("inf")))

term("tflops_hdim128_seq8k",
    col("tflops_hdim128_seq8k").higher().id()
    .sigmoid())

term("latency_ms_hdim128_seq8k",
    col("latency_ms_hdim128_seq8k").lower().id()
    .bounded_power(bound=0.0))

term("max_diff_hdim128_seq8k",
    col("max_diff_hdim128_seq8k").lower().id()
    .bounded_power(bound=0.0))

term("correct_hdim128_seq8k",
    penalty_lower(col("correct_hdim128_seq8k").higher().id(),
                  target=1.0, sharpness=float("inf")))

term("tflops_hdim256_seq16k",
    col("tflops_hdim256_seq16k").higher().id()
    .sigmoid())

term("latency_ms_hdim256_seq16k",
    col("latency_ms_hdim256_seq16k").lower().id()
    .bounded_power(bound=0.0))

term("max_diff_hdim256_seq16k",
    col("max_diff_hdim256_seq16k").lower().id()
    .bounded_power(bound=0.0))

term("correct_hdim256_seq16k",
    penalty_lower(col("correct_hdim256_seq16k").higher().id(),
                  target=1.0, sharpness=float("inf")))

term("speedup_vs_sdpa_hdim64_seq4k",
    col("speedup_vs_sdpa_hdim64_seq4k").higher().id()
    .sigmoid())

term("speedup_vs_sdpa_hdim128_seq8k",
    col("speedup_vs_sdpa_hdim128_seq8k").higher().id()
    .sigmoid())

term("speedup_vs_sdpa_hdim256_seq16k",
    col("speedup_vs_sdpa_hdim256_seq16k").higher().id()
    .sigmoid())

setting("hdim64_seq4k",
    weighted_mean(
        ("tflops_hdim64_seq4k", 1.0),
        ("latency_ms_hdim64_seq4k", 1.0),
        ("max_diff_hdim64_seq4k", 1.0),
        ("speedup_vs_sdpa_hdim64_seq4k", 1.0),
    ),
    constraints=["correct_hdim64_seq4k"],
)
setting("hdim128_seq8k",
    weighted_mean(
        ("tflops_hdim128_seq8k", 1.0),
        ("latency_ms_hdim128_seq8k", 1.0),
        ("max_diff_hdim128_seq8k", 1.0),
        ("speedup_vs_sdpa_hdim128_seq8k", 1.0),
    ),
    constraints=["correct_hdim128_seq8k"],
)
setting("hdim256_seq16k",
    weighted_mean(
        ("tflops_hdim256_seq16k", 1.0),
        ("latency_ms_hdim256_seq16k", 1.0),
        ("max_diff_hdim256_seq16k", 1.0),
        ("speedup_vs_sdpa_hdim256_seq16k", 1.0),
    ),
    constraints=["correct_hdim256_seq16k"],
)

task(gmean("hdim64_seq4k", "hdim128_seq8k", "hdim256_seq16k"))
