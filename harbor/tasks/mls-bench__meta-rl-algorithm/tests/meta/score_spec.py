"""Score spec for meta-rl-algorithm (auto-generated, review before use)."""
from mlsbench.scoring.dsl import *

term("meta_test_return_point_robot",
    col("meta_test_return_point_robot").higher().id()
    .sigmoid())

term("meta_test_return_cheetah_vel",
    col("meta_test_return_cheetah_vel").higher().id()
    .sigmoid())

term("meta_test_return_sparse_point_robot",
    col("meta_test_return_sparse_point_robot").higher().id()
    .sigmoid())

setting("point-robot", weighted_mean(("meta_test_return_point_robot", 1.0)))
setting("cheetah-vel", weighted_mean(("meta_test_return_cheetah_vel", 1.0)))
setting("sparse-point-robot", weighted_mean(("meta_test_return_sparse_point_robot", 1.0)))

task(gmean("point-robot", "cheetah-vel", "sparse-point-robot"))
