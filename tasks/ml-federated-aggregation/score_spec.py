"""Score spec for ml-federated-aggregation."""
from mlsbench.scoring.dsl import *

# accuracy is on 0-1 scale based on leaderboard values (0.6338, 0.4851, 0.8108)
# best_accuracy_* is not scored. The task's metric is the test accuracy after
# the last round; best_accuracy is the maximum over the every-10-round test
# evaluations, i.e. checkpoint selection on the test set. For every baseline
# on cifar10 and shakespeare it equals test_accuracy exactly (0.5954/0.5954,
# 0.436/0.436, ...), so scoring both counted final accuracy twice.

term("test_accuracy_cifar10",
    col("test_accuracy_cifar10").higher().id()
    .bounded_power(bound=1.0))

term("test_loss_cifar10",
    col("test_loss_cifar10").lower().id()
    .bounded_power(bound=0.0))

term("test_accuracy_shakespeare",
    col("test_accuracy_shakespeare").higher().id()
    .bounded_power(bound=1.0))

term("test_loss_shakespeare",
    col("test_loss_shakespeare").lower().id()
    .bounded_power(bound=0.0))

term("test_accuracy_femnist",
    col("test_accuracy_femnist").higher().id()
    .bounded_power(bound=1.0))

term("test_loss_femnist",
    col("test_loss_femnist").lower().id()
    .bounded_power(bound=0.0))

setting("cifar10", weighted_mean(
    ("test_accuracy_cifar10", 1.0),
    ("test_loss_cifar10", 1.0),
))
setting("shakespeare", weighted_mean(
    ("test_accuracy_shakespeare", 1.0),
    ("test_loss_shakespeare", 1.0),
))
setting("femnist", weighted_mean(
    ("test_accuracy_femnist", 1.0),
    ("test_loss_femnist", 1.0),
))

task(gmean("cifar10", "shakespeare", "femnist"))
