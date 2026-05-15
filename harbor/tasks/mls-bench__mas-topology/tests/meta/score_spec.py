"""Score spec for mas-topology.

Three evaluation settings:
  - humaneval-4-deepseek: MacNet 4-agent on HumanEval with deepseek-chat backend
  - humaneval-4-qwen:     MacNet 4-agent on HumanEval with qwen2.5-72b-instruct backend
  - srdd-4-deepseek:      MacNet 4-agent on SRDD with deepseek-chat backend

Reference point: chain baseline (simplest non-trivial topology).
Count columns (passed/total/srdd_passed/srdd_total/mean_loc) and elapsed_* are
informational only and do not enter scoring.
"""
from mlsbench.scoring.dsl import *

term("pass_at_1_deepseek",
    col("pass_at_1_deepseek").higher().id()
    .bounded_power(bound=1.0))

term("pass_at_1_qwen",
    col("pass_at_1_qwen").higher().id()
    .bounded_power(bound=1.0))

term("srdd_exec_rate",
    col("srdd_exec_rate").higher().id()
    .bounded_power(bound=1.0))

setting("humaneval-4-deepseek", weighted_mean(("pass_at_1_deepseek", 1.0)))
setting("humaneval-4-qwen",     weighted_mean(("pass_at_1_qwen", 1.0)))
setting("srdd-4-deepseek",      weighted_mean(("srdd_exec_rate", 1.0)))

task(gmean("humaneval-4-deepseek", "humaneval-4-qwen", "srdd-4-deepseek"))
