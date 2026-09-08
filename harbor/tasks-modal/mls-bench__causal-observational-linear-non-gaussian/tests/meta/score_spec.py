"""Score spec for causal-observational-linear-non-gaussian."""
from mlsbench.scoring.dsl import *

# Current config labels: ER30, ER50, SF100 (hidden).

term("shd_ER30",
    col("shd_ER30").lower().id()
    .bounded_power(bound=0.0))

term("f1_ER30",
    col("f1_ER30").higher().id()
    .bounded_power(bound=1.0))

term("precision_ER30",
    col("precision_ER30").higher().id()
    .bounded_power(bound=1.0))

term("recall_ER30",
    col("recall_ER30").higher().id()
    .bounded_power(bound=1.0))

term("shd_ER50",
    col("shd_ER50").lower().id()
    .bounded_power(bound=0.0))

term("f1_ER50",
    col("f1_ER50").higher().id()
    .bounded_power(bound=1.0))

term("precision_ER50",
    col("precision_ER50").higher().id()
    .bounded_power(bound=1.0))

term("recall_ER50",
    col("recall_ER50").higher().id()
    .bounded_power(bound=1.0))

term("shd_SF100",
    col("shd_SF100").lower().id()
    .bounded_power(bound=0.0))

term("f1_SF100",
    col("f1_SF100").higher().id()
    .bounded_power(bound=1.0))

term("precision_SF100",
    col("precision_SF100").higher().id()
    .bounded_power(bound=1.0))

term("recall_SF100",
    col("recall_SF100").higher().id()
    .bounded_power(bound=1.0))

setting("ER30", weighted_mean(("shd_ER30", 1.0), ("f1_ER30", 1.0), ("precision_ER30", 1.0), ("recall_ER30", 1.0)))
setting("ER50", weighted_mean(("shd_ER50", 1.0), ("f1_ER50", 1.0), ("precision_ER50", 1.0), ("recall_ER50", 1.0)))
setting("SF100", weighted_mean(("shd_SF100", 1.0), ("f1_SF100", 1.0), ("precision_SF100", 1.0), ("recall_SF100", 1.0)))

task(gmean("ER30", "ER50", "SF100"))
