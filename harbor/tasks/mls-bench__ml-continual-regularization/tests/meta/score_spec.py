"""Score spec for ml-continual-regularization."""
from mlsbench.scoring.dsl import *

# accuracy is on 0-1 scale based on leaderboard values
# Using average_accuracy as the summary metric instead of all per-context accuracies

term("average_accuracy_split_mnist",
    col("average_accuracy_split_mnist").higher().id()
    .bounded_power(bound=1.0))

term("average_accuracy_perm_mnist",
    col("average_accuracy_perm_mnist").higher().id()
    .bounded_power(bound=1.0))

term("average_accuracy_split_cifar100",
    col("average_accuracy_split_cifar100").higher().id()
    .bounded_power(bound=1.0))

setting("split-mnist", weighted_mean(("average_accuracy_split_mnist", 1.0)))
setting("perm-mnist", weighted_mean(("average_accuracy_perm_mnist", 1.0)))
setting("split-cifar100", weighted_mean(("average_accuracy_split_cifar100", 1.0)))

task(gmean("split-mnist", "perm-mnist", "split-cifar100"))
