"""Score spec for llm-offline-rl."""
from mlsbench.scoring.dsl import *

term("gsm8k_accuracy",
    col("gsm8k_accuracy").higher().id()
    .bounded_power(bound=100.0))

term("math500_accuracy",
    col("math500_accuracy").higher().id()
    .bounded_power(bound=100.0))

term("aime2024_accuracy",
    col("aime2024_accuracy").higher().id()
    .bounded_power(bound=100.0))

setting("math_eval", weighted_mean(
    ("gsm8k_accuracy", 1.0),
    ("math500_accuracy", 1.0),
    ("aime2024_accuracy", 1.0),
))

task(gmean("math_eval"))
