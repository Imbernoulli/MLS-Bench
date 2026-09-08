"""Score spec for ts-anomaly-detection (auto-generated, review before use)."""
from mlsbench.scoring.dsl import *

term("f_score_PSM",
    col("f_score_PSM").higher().id()
    .bounded_power(bound=1.0))

term("precision_PSM",
    col("precision_PSM").higher().id()
    .bounded_power(bound=1.0))

term("recall_PSM",
    col("recall_PSM").higher().id()
    .bounded_power(bound=1.0))

term("f_score_MSL",
    col("f_score_MSL").higher().id()
    .bounded_power(bound=1.0))

term("precision_MSL",
    col("precision_MSL").higher().id()
    .bounded_power(bound=1.0))

term("recall_MSL",
    col("recall_MSL").higher().id()
    .bounded_power(bound=1.0))

term("f_score_SMAP",
    col("f_score_SMAP").higher().id()
    .bounded_power(bound=1.0))

term("precision_SMAP",
    col("precision_SMAP").higher().id()
    .bounded_power(bound=1.0))

term("recall_SMAP",
    col("recall_SMAP").higher().id()
    .bounded_power(bound=1.0))

setting("PSM", weighted_mean(("f_score_PSM", 1.0), ("precision_PSM", 1.0), ("recall_PSM", 1.0)))
setting("MSL", weighted_mean(("f_score_MSL", 1.0), ("precision_MSL", 1.0), ("recall_MSL", 1.0)))
setting("SMAP", weighted_mean(("f_score_SMAP", 1.0), ("precision_SMAP", 1.0), ("recall_SMAP", 1.0)))

task(gmean("PSM", "MSL", "SMAP"))
