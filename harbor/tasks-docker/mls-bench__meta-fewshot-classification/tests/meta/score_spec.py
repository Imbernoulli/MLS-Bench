"""Score spec for meta-fewshot-classification."""
from mlsbench.scoring.dsl import *

# accuracy is on 0-1 scale based on leaderboard values (0.6991, 0.8044, 0.7578)
term("accuracy_mini_imagenet",
    col("accuracy_mini_imagenet").higher().id()
    .bounded_power(bound=1.0))

term("accuracy_cifar_fs",
    col("accuracy_cifar_fs").higher().id()
    .bounded_power(bound=1.0))

term("accuracy_CUB",
    col("accuracy_CUB").higher().id()
    .bounded_power(bound=1.0))

setting("mini_imagenet", weighted_mean(("accuracy_mini_imagenet", 1.0)))
setting("cifar_fs", weighted_mean(("accuracy_cifar_fs", 1.0)))
setting("CUB", weighted_mean(("accuracy_CUB", 1.0)))

task(gmean("mini_imagenet", "cifar_fs", "CUB"))
