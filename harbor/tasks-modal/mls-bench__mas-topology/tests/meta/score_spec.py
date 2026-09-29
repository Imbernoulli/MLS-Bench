"""Score spec for mas-topology.

Three evaluation settings:
  - humaneval-4-deepseek: MacNet 4-agent on HumanEval with deepseek-chat backend
  - humaneval-4-qwen:     MacNet 4-agent on HumanEval with qwen2.5-72b-instruct backend
  - srdd-4-deepseek:      MacNet 4-agent on SRDD with deepseek-chat backend

Count columns (passed/total/srdd_passed/srdd_total/mean_loc) and elapsed_* are
informational only and do not enter scoring.

Anchors. Every term's floor is the weakest baseline for that metric
(``BaselineAnchors.worst_for``). The rates are counts over small sets: 33
HumanEval problems (one problem = 0.030) and 20 SRDD prompts (one prompt =
0.05). The baselines differ by 3 and 5 problems on HumanEval and by 2 prompts
on SRDD, which is one to two binomial standard deviations. Calibrating each
term on that spread (the default, best baseline = 0.5) made one SRDD prompt
swing the task score by about 0.3. Every term instead uses a fixed sigmoid
scale, set so that a stated gain over the weakest baseline scores 0.5
(2 * sigmoid(gain / scale) - 1 = 0.5, i.e. scale = gain / ln 3):

- HumanEval pass@1: +0.33 (11 of 33 problems).
- SRDD executability rate: +0.25 (5 of 20 prompts).

Each gain is about four binomial standard deviations at the baselines' rates
(0.08 on HumanEval at pass@1 ~0.7, 0.062 on SRDD at ~0.08). A 1-sigma
difference moves a term by at most about 0.14, and real gains still register.
"""
import math

from mlsbench.scoring.dsl import *

# 2 * sigmoid(gain / scale) - 1 = 0.5  <=>  scale = gain / ln 3.
SCALE_HUMANEVAL = 0.33 / math.log(3.0)  # +11/33 problems -> 0.5
SCALE_SRDD = 0.25 / math.log(3.0)       # +5/20 prompts -> 0.5

term("pass_at_1_deepseek",
    col("pass_at_1_deepseek").higher().id()
    .sigmoid(scale=SCALE_HUMANEVAL))

term("pass_at_1_qwen",
    col("pass_at_1_qwen").higher().id()
    .sigmoid(scale=SCALE_HUMANEVAL))

term("srdd_exec_rate",
    col("srdd_exec_rate").higher().id()
    .sigmoid(scale=SCALE_SRDD))

setting("humaneval-4-deepseek", weighted_mean(("pass_at_1_deepseek", 1.0)))
setting("humaneval-4-qwen",     weighted_mean(("pass_at_1_qwen", 1.0)))
setting("srdd-4-deepseek",      weighted_mean(("srdd_exec_rate", 1.0)))

task(gmean("humaneval-4-deepseek", "humaneval-4-qwen", "srdd-4-deepseek"))
