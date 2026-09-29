"""Score spec for optimization-dp-sgd."""
from mlsbench.scoring.dsl import *

# accuracy on 0-100 scale; best_accuracy (peak test accuracy during training)
# is the scored metric, test_accuracy (final epoch) is redundant with it.
# epsilon_* is recorded but not scored: the fixed harness aborts any run
# whose accounted epsilon exceeds the target budget (epsilon = 3), so every
# scored run is at the same privacy level and only accuracy is compared.
# Rewarding a lower epsilon would pay for under-spending the fixed budget.

term("best_accuracy_mnist",
    col("best_accuracy_mnist").higher().id()
    .bounded_power(bound=100.0))

term("best_accuracy_fmnist",
    col("best_accuracy_fmnist").higher().id()
    .bounded_power(bound=100.0))

term("best_accuracy_cifar10",
    col("best_accuracy_cifar10").higher().id()
    .bounded_power(bound=100.0))

setting("mnist", weighted_mean(
    ("best_accuracy_mnist", 1.0),
))
setting("fmnist", weighted_mean(
    ("best_accuracy_fmnist", 1.0),
))
setting("cifar10", weighted_mean(
    ("best_accuracy_cifar10", 1.0),
))

task(gmean("mnist", "fmnist", "cifar10"))
