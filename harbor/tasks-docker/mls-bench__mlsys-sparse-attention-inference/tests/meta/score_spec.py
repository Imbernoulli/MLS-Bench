"""Score spec for mlsys-sparse-attention-inference.

Systems task: each env contributes both a quality score and a sparsity
score; the task score is the gmean across the 3 envs so an agent has to
do well on all of them, not just average up.

Wall-clock time is intentionally NOT scored. Pure-PyTorch sparse
implementations have non-trivial overhead from gather/scatter and mask
construction, so they are typically slower than fused SDPA dense; scoring
on time would penalize the very methods this task is studying. Density
(the architectural FLOPs proxy) is the relevant efficiency axis here.
"""
from mlsbench.scoring.dsl import *

# ── Per-env quality terms ──
term("niah_acc",
    col("niah_acc").higher().id()
    .bounded_power(bound=1.0))
term("qasper_f1",
    col("qasper_f1").higher().id()
    .bounded_power(bound=1.0))
term("multifieldqa_f1",
    col("multifieldqa_f1").higher().id()
    .bounded_power(bound=1.0))

# ── Per-env density terms (lower density = sparser = better) ──
term("niah_density",
    col("niah_density").lower().id()
    .bounded_power(bound=0.0))
term("qasper_density",
    col("qasper_density").lower().id()
    .bounded_power(bound=0.0))
term("multifieldqa_density",
    col("multifieldqa_density").lower().id()
    .bounded_power(bound=0.0))

# ── Per-env settings: 70% quality, 30% sparsity ──
setting("niah_8k",
    weighted_mean(("niah_acc", 0.7), ("niah_density", 0.3)))
setting("longbench_qasper",
    weighted_mean(("qasper_f1", 0.7), ("qasper_density", 0.3)))
setting("longbench_multifieldqa_en",
    weighted_mean(("multifieldqa_f1", 0.7), ("multifieldqa_density", 0.3)))

task(gmean("niah_8k", "longbench_qasper", "longbench_multifieldqa_en"))
