"""Baseline-free official GSM8K accuracy on its natural [0, 1] range."""
from mlsbench.scoring.dsl import *

term(
    "accuracy_gsm8k",
    col("accuracy_gsm8k").higher().id().bounded_power(
        bound=1.0,
        floor=const(0.0),
    ),
)
setting("gsm8k", weighted_mean(("accuracy_gsm8k", 1.0)))
task(gmean("gsm8k"))
