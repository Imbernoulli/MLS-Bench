"""Score spec for optimization-dp-sgd."""
from mlsbench.scoring.dsl import *

# accuracy on 0-100 scale (values: 95.78, 80.68, 61.25)
# epsilon: lower is better (less privacy budget consumption), bound=0.0
# best_accuracy: peak accuracy during training — keep as the primary accuracy metric
# test_accuracy: final epoch accuracy — dropped as redundant with best_accuracy
# refs from best baseline (automatic_clipping / adaptive_clipping) means

term("best_accuracy_mnist",
    col("best_accuracy_mnist").higher().id()
    .bounded_power(bound=100.0))

term("epsilon_mnist",
    col("epsilon_mnist").lower().id()
    .bounded_power(bound=0.0))

term("best_accuracy_fmnist",
    col("best_accuracy_fmnist").higher().id()
    .bounded_power(bound=100.0))

term("epsilon_fmnist",
    col("epsilon_fmnist").lower().id()
    .bounded_power(bound=0.0))

term("best_accuracy_cifar10",
    col("best_accuracy_cifar10").higher().id()
    .bounded_power(bound=100.0))

term("epsilon_cifar10",
    col("epsilon_cifar10").lower().id()
    .bounded_power(bound=0.0))

setting("mnist", weighted_mean(
    ("best_accuracy_mnist", 1.0),
    ("epsilon_mnist", 1.0),
))
setting("fmnist", weighted_mean(
    ("best_accuracy_fmnist", 1.0),
    ("epsilon_fmnist", 1.0),
))
setting("cifar10", weighted_mean(
    ("best_accuracy_cifar10", 1.0),
    ("epsilon_cifar10", 1.0),
))

task(gmean("mnist", "fmnist", "cifar10"))
