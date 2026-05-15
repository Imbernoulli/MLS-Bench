"""Score spec for jepa-prediction-loss.

mean_detection_ap is bounded in [0, 1] (higher is better). MSE is the
standard / default prediction loss for JEPA, so its mean across model
sizes serves as the reference point that maps to score 0.5.
"""
from mlsbench.scoring.dsl import *

term("mean_detection_ap_small",
    col("mean_detection_ap_small").higher().id()
    .bounded_power(bound=1.0))

term("mean_detection_ap_base",
    col("mean_detection_ap_base").higher().id()
    .bounded_power(bound=1.0))

term("mean_detection_ap_large",
    col("mean_detection_ap_large").higher().id()
    .bounded_power(bound=1.0))

setting("small", weighted_mean(("mean_detection_ap_small", 1.0)))
setting("base",  weighted_mean(("mean_detection_ap_base", 1.0)))
setting("large", weighted_mean(("mean_detection_ap_large", 1.0)))

task(gmean("small", "base", "large"))
