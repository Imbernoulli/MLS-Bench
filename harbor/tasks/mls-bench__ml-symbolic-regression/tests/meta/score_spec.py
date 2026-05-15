"""Score spec for ml-symbolic-regression (auto-generated, review before use)."""
from mlsbench.scoring.dsl import *

term("test_r2_nguyen7",
    col("test_r2_nguyen7").higher().id()
    .bounded_power(bound=1.0))

term("test_r2_nguyen10",
    col("test_r2_nguyen10").higher().id()
    .bounded_power(bound=1.0))

term("test_r2_koza3",
    col("test_r2_koza3").higher().id()
    .bounded_power(bound=1.0))

setting("nguyen7", weighted_mean(("test_r2_nguyen7", 1.0)))
setting("nguyen10", weighted_mean(("test_r2_nguyen10", 1.0)))
setting("koza3", weighted_mean(("test_r2_koza3", 1.0)))

task(gmean("nguyen7", "nguyen10", "koza3"))
