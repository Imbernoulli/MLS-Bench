"""Score spec for stf-traffic-forecast (auto-generated, review before use)."""
from mlsbench.scoring.dsl import *

term("mae_METR_LA",
    col("mae_METR-LA").lower().id()
    .bounded_power(bound=0.0))

term("rmse_METR_LA",
    col("rmse_METR-LA").lower().id()
    .bounded_power(bound=0.0))

term("mape_METR_LA",
    col("mape_METR-LA").lower().id()
    .bounded_power(bound=0.0))

term("mae_PEMS_BAY",
    col("mae_PEMS-BAY").lower().id()
    .bounded_power(bound=0.0))

term("rmse_PEMS_BAY",
    col("rmse_PEMS-BAY").lower().id()
    .bounded_power(bound=0.0))

term("mape_PEMS_BAY",
    col("mape_PEMS-BAY").lower().id()
    .bounded_power(bound=0.0))

term("mae_PEMS04",
    col("mae_PEMS04").lower().id()
    .bounded_power(bound=0.0))

term("rmse_PEMS04",
    col("rmse_PEMS04").lower().id()
    .bounded_power(bound=0.0))

term("mape_PEMS04",
    col("mape_PEMS04").lower().id()
    .bounded_power(bound=0.0))

setting("METR-LA", weighted_mean(("mae_METR_LA", 1.0), ("rmse_METR_LA", 1.0), ("mape_METR_LA", 1.0)))
setting("PEMS-BAY", weighted_mean(("mae_PEMS_BAY", 1.0), ("rmse_PEMS_BAY", 1.0), ("mape_PEMS_BAY", 1.0)))
setting("PEMS04", weighted_mean(("mae_PEMS04", 1.0), ("rmse_PEMS04", 1.0), ("mape_PEMS04", 1.0)))

task(gmean("METR-LA", "PEMS-BAY", "PEMS04"))
