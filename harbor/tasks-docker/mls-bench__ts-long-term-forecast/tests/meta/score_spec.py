"""Score spec for ts-long-term-forecast (auto-generated, review before use)."""
from mlsbench.scoring.dsl import *

term("mse_ETTh1",
    col("mse_ETTh1").lower().id()
    .bounded_power(bound=0.0))

term("mae_ETTh1",
    col("mae_ETTh1").lower().id()
    .bounded_power(bound=0.0))

term("mse_Weather",
    col("mse_Weather").lower().id()
    .bounded_power(bound=0.0))

term("mae_Weather",
    col("mae_Weather").lower().id()
    .bounded_power(bound=0.0))

term("mse_ECL",
    col("mse_ECL").lower().id()
    .bounded_power(bound=0.0))

term("mae_ECL",
    col("mae_ECL").lower().id()
    .bounded_power(bound=0.0))

setting("ETTh1", weighted_mean(("mse_ETTh1", 1.0), ("mae_ETTh1", 1.0)))
setting("Weather", weighted_mean(("mse_Weather", 1.0), ("mae_Weather", 1.0)))
setting("ECL", weighted_mean(("mse_ECL", 1.0), ("mae_ECL", 1.0)))

task(gmean("ETTh1", "Weather", "ECL"))
