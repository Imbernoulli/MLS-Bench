"""Score spec for causal-observational-nonlinear."""
from mlsbench.scoring.dsl import *

# Current config labels: SF20-GP, ER20-Gauss, ER12-LowSample (hidden).

term("shd_SF20_GP",
    col("shd_SF20-GP").lower().id()
    .bounded_power(bound=0.0))

term("f1_SF20_GP",
    col("f1_SF20-GP").higher().id()
    .bounded_power(bound=1.0))

term("precision_SF20_GP",
    col("precision_SF20-GP").higher().id()
    .bounded_power(bound=1.0))

term("recall_SF20_GP",
    col("recall_SF20-GP").higher().id()
    .bounded_power(bound=1.0))

term("shd_ER20_Gauss",
    col("shd_ER20-Gauss").lower().id()
    .bounded_power(bound=0.0))

term("f1_ER20_Gauss",
    col("f1_ER20-Gauss").higher().id()
    .bounded_power(bound=1.0))

term("precision_ER20_Gauss",
    col("precision_ER20-Gauss").higher().id()
    .bounded_power(bound=1.0))

term("recall_ER20_Gauss",
    col("recall_ER20-Gauss").higher().id()
    .bounded_power(bound=1.0))

term("shd_ER12_LowSample",
    col("shd_ER12-LowSample").lower().id()
    .bounded_power(bound=0.0))

term("f1_ER12_LowSample",
    col("f1_ER12-LowSample").higher().id()
    .bounded_power(bound=1.0))

term("precision_ER12_LowSample",
    col("precision_ER12-LowSample").higher().id()
    .bounded_power(bound=1.0))

term("recall_ER12_LowSample",
    col("recall_ER12-LowSample").higher().id()
    .bounded_power(bound=1.0))

setting("ER12-LowSample", weighted_mean(("shd_ER12_LowSample", 1.0), ("f1_ER12_LowSample", 1.0), ("precision_ER12_LowSample", 1.0), ("recall_ER12_LowSample", 1.0)))
setting("SF20-GP", weighted_mean(("shd_SF20_GP", 1.0), ("f1_SF20_GP", 1.0), ("precision_SF20_GP", 1.0), ("recall_SF20_GP", 1.0)))
setting("ER20-Gauss", weighted_mean(("shd_ER20_Gauss", 1.0), ("f1_ER20_Gauss", 1.0), ("precision_ER20_Gauss", 1.0), ("recall_ER20_Gauss", 1.0)))

task(gmean("SF20-GP", "ER20-Gauss", "ER12-LowSample"))
