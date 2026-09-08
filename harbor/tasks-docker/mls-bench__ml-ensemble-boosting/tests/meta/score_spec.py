"""Score spec for ml-ensemble-boosting (auto-generated, review before use)."""
from mlsbench.scoring.dsl import *

term("test_accuracy_breast_cancer",
    col("test_accuracy_breast_cancer").higher().id()
    .bounded_power(bound=1.0))

term("test_rmse_diabetes",
    col("test_rmse_diabetes").lower().id()
    .bounded_power(bound=0.0))

term("test_rmse_california_housing",
    col("test_rmse_california_housing").lower().id()
    .bounded_power(bound=0.0))

setting("breast_cancer", weighted_mean(("test_accuracy_breast_cancer", 1.0)))
setting("diabetes", weighted_mean(("test_rmse_diabetes", 1.0)))
setting("california_housing", weighted_mean(("test_rmse_california_housing", 1.0)))

task(gmean("breast_cancer", "diabetes", "california_housing"))
