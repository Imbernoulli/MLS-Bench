"""Baseline-free scoring for complete official summarization test splits.

Mean per-example ROUGE-L F1 has a fixed [0, 1] range. Zero overlap maps to zero,
a mean F1 of 0.30 maps to 0.5, and perfect overlap maps to 1. This mapping is
shared by all ten repo siblings and does not read a fabricated or fallback
leaderboard value. The verifier separately requires every setting, inventory,
model-or-source, metric, and completion proof; any failed verification receives
exact 0.
"""
from mlsbench.scoring.dsl import *


def _rouge_term(metric: str):
    return (
        col(metric)
        .higher()
        .id()
        .bounded_power(
            bound=1.0,
            floor=const(0.0),
            ref=const(0.30),
            ref_score=0.5,
        )
    )


for _setting in ("xsum", "cnndm", "samsum"):
    _metric = f"rougeL_{_setting}"
    term(_metric, _rouge_term(_metric))
    setting(_setting, weighted_mean((_metric, 1.0)))

task(gmean("xsum", "cnndm", "samsum"))
