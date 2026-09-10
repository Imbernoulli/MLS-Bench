"""Score spec for marl-centralized-critic (auto-generated)."""
from mlsbench.scoring.dsl import *

term("test_return_mean_mmm",
    col("test_return_mean_mmm").higher().id()
    .sigmoid())

term("test_battle_won_mean_mmm",
    col("test_battle_won_mean_mmm").higher().id()
    .bounded_power(bound=1.0))

term("test_return_mean_2s3z",
    col("test_return_mean_2s3z").higher().id()
    .sigmoid())

term("test_battle_won_mean_2s3z",
    col("test_battle_won_mean_2s3z").higher().id()
    .bounded_power(bound=1.0))

term("test_return_mean_3s5z",
    col("test_return_mean_3s5z").higher().id()
    .sigmoid())

term("test_battle_won_mean_3s5z",
    col("test_battle_won_mean_3s5z").higher().id()
    .bounded_power(bound=1.0))

setting("mmm", weighted_mean(("test_return_mean_mmm", 1.0), ("test_battle_won_mean_mmm", 1.0)))
setting("2s3z", weighted_mean(("test_return_mean_2s3z", 1.0), ("test_battle_won_mean_2s3z", 1.0)))
setting("3s5z", weighted_mean(("test_return_mean_3s5z", 1.0), ("test_battle_won_mean_3s5z", 1.0)))
task(gmean("mmm", "2s3z", "3s5z"))
