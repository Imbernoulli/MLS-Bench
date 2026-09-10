"""Score spec for ts-short-term-forecast (auto-generated, review before use)."""
from mlsbench.scoring.dsl import *

term("smape_m4_monthly",
    col("smape_m4_monthly").lower().id()
    .bounded_power(bound=0.0))

term("mape_m4_monthly",
    col("mape_m4_monthly").lower().id()
    .bounded_power(bound=0.0))

term("smape_m4_quarterly",
    col("smape_m4_quarterly").lower().id()
    .bounded_power(bound=0.0))

term("mape_m4_quarterly",
    col("mape_m4_quarterly").lower().id()
    .bounded_power(bound=0.0))

term("smape_m4_yearly",
    col("smape_m4_yearly").lower().id()
    .bounded_power(bound=0.0))

term("mape_m4_yearly",
    col("mape_m4_yearly").lower().id()
    .bounded_power(bound=0.0))

setting("m4_monthly", weighted_mean(("smape_m4_monthly", 1.0), ("mape_m4_monthly", 1.0)))
setting("m4_quarterly", weighted_mean(("smape_m4_quarterly", 1.0), ("mape_m4_quarterly", 1.0)))
setting("m4_yearly", weighted_mean(("smape_m4_yearly", 1.0), ("mape_m4_yearly", 1.0)))

task(gmean("m4_monthly", "m4_quarterly", "m4_yearly"))
