"""Score spec for ml-missing-data-imputation."""
from mlsbench.scoring.dsl import *

# rmse: lower is better, bound=0.0 (no negative error)
# downstream_score: higher is better, on 0-1 scale (values ~0.93-0.96)
term("rmse_breast_cancer",
    col("rmse_breast_cancer").lower().id()
    .bounded_power(bound=0.0))

term("downstream_score_breast_cancer",
    col("downstream_score_breast_cancer").higher().id()
    .bounded_power(bound=1.0))

term("rmse_wine",
    col("rmse_wine").lower().id()
    .bounded_power(bound=0.0))

term("downstream_score_wine",
    col("downstream_score_wine").higher().id()
    .bounded_power(bound=1.0))

term("rmse_california",
    col("rmse_california").lower().id()
    .bounded_power(bound=0.0))

term("downstream_score_california",
    col("downstream_score_california").higher().id()
    .bounded_power(bound=1.0))

setting("breast_cancer", weighted_mean(
    ("rmse_breast_cancer", 1.0),
    ("downstream_score_breast_cancer", 1.0),
))
setting("wine", weighted_mean(
    ("rmse_wine", 1.0),
    ("downstream_score_wine", 1.0),
))
setting("california", weighted_mean(
    ("rmse_california", 1.0),
    ("downstream_score_california", 1.0),
))

task(gmean("breast_cancer", "wine", "california"))
