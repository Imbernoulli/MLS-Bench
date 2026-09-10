"""Score spec for llm-rl-reward-normalization.

Primary metric = arithmetic mean of three math-reasoning benchmark accuracies
(GSM8K, MATH-500, AMC 2022-2023) after 100 steps of GRPO fine-tuning on
Qwen2.5-0.5B, with a customisable upstream reward-normalization step.

Normalization uses dynamic leaderboard anchors: the worst baseline is the
0-point floor and the best baseline is the 50-point anchor.
"""
from mlsbench.scoring.dsl import *

term("gsm8k",
    col("gsm8k_accuracy").higher().id()
    .bounded_power(bound=1.0))

term("math500",
    col("math500_accuracy").higher().id()
    .bounded_power(bound=1.0))

term("amc",
    col("amc_accuracy").higher().id()
    .bounded_power(bound=1.0))

setting("qwen2.5-0.5b", weighted_mean(
    ("gsm8k", 1.0),
    ("math500", 1.0),
    ("amc", 1.0),
))

task(gmean("qwen2.5-0.5b"))
