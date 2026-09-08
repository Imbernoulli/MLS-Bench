"""Score spec for meta-inner-loop-optimizer."""
from mlsbench.scoring.dsl import *

# accuracy is on 0-1 scale based on leaderboard values (0.4573, 0.6462, 0.71)
term("accuracy_mini_imagenet_1shot",
    col("accuracy_mini_imagenet_1shot").higher().id()
    .bounded_power(bound=1.0))

term("accuracy_mini_imagenet_5shot",
    col("accuracy_mini_imagenet_5shot").higher().id()
    .bounded_power(bound=1.0))

term("accuracy_cifar_fs_5shot",
    col("accuracy_cifar_fs_5shot").higher().id()
    .bounded_power(bound=1.0))

setting("mini_imagenet_1shot", weighted_mean(("accuracy_mini_imagenet_1shot", 1.0)))
setting("mini_imagenet_5shot", weighted_mean(("accuracy_mini_imagenet_5shot", 1.0)))
setting("cifar_fs_5shot", weighted_mean(("accuracy_cifar_fs_5shot", 1.0)))

task(gmean("mini_imagenet_1shot", "mini_imagenet_5shot", "cifar_fs_5shot"))
