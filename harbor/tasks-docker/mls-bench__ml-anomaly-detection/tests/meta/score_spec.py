"""Score spec for ml-anomaly-detection (auto-generated, review before use)."""
from mlsbench.scoring.dsl import *

term("auroc_cardio",
    col("auroc_cardio").higher().id()
    .bounded_power(bound=1.0))

term("f1_cardio",
    col("f1_cardio").higher().id()
    .bounded_power(bound=1.0))

term("auroc_thyroid",
    col("auroc_thyroid").higher().id()
    .bounded_power(bound=1.0))

term("f1_thyroid",
    col("f1_thyroid").higher().id()
    .bounded_power(bound=1.0))

term("auroc_satellite",
    col("auroc_satellite").higher().id()
    .bounded_power(bound=1.0))

term("f1_satellite",
    col("f1_satellite").higher().id()
    .bounded_power(bound=1.0))

term("auroc_shuttle",
    col("auroc_shuttle").higher().id()
    .bounded_power(bound=1.0))

term("f1_shuttle",
    col("f1_shuttle").higher().id()
    .bounded_power(bound=1.0))

setting("cardio", weighted_mean(("auroc_cardio", 1.0), ("f1_cardio", 1.0)))
setting("thyroid", weighted_mean(("auroc_thyroid", 1.0), ("f1_thyroid", 1.0)))
setting("satellite", weighted_mean(("auroc_satellite", 1.0), ("f1_satellite", 1.0)))
setting("shuttle", weighted_mean(("auroc_shuttle", 1.0), ("f1_shuttle", 1.0)))

task(gmean("cardio", "thyroid", "satellite", "shuttle"))
