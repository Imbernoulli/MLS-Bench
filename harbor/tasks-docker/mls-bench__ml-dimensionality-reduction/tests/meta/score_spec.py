"""Score spec for ml-dimensionality-reduction."""
from mlsbench.scoring.dsl import *

# knn_acc: 0-1 scale based on leaderboard values (0.862, 0.790, 0.687)
# trustworthiness, continuity: 0-1 scale, higher is better
# fashion_mnist has its own test_cmd label — separate from mnist setting

term("knn_acc_mnist",
    col("knn_acc_mnist").higher().id()
    .bounded_power(bound=1.0))

term("trustworthiness_mnist",
    col("trustworthiness_mnist").higher().id()
    .bounded_power(bound=1.0))

term("continuity_mnist",
    col("continuity_mnist").higher().id()
    .bounded_power(bound=1.0))

term("knn_acc_fashion_mnist",
    col("knn_acc_fashion_mnist").higher().id()
    .bounded_power(bound=1.0))

term("trustworthiness_fashion_mnist",
    col("trustworthiness_fashion_mnist").higher().id()
    .bounded_power(bound=1.0))

term("continuity_fashion_mnist",
    col("continuity_fashion_mnist").higher().id()
    .bounded_power(bound=1.0))

term("knn_acc_newsgroups",
    col("knn_acc_newsgroups").higher().id()
    .bounded_power(bound=1.0))

term("trustworthiness_newsgroups",
    col("trustworthiness_newsgroups").higher().id()
    .bounded_power(bound=1.0))

term("continuity_newsgroups",
    col("continuity_newsgroups").higher().id()
    .bounded_power(bound=1.0))

setting("mnist", weighted_mean(
    ("knn_acc_mnist", 1.0),
    ("trustworthiness_mnist", 1.0),
    ("continuity_mnist", 1.0),
))
setting("fashion_mnist", weighted_mean(
    ("knn_acc_fashion_mnist", 1.0),
    ("trustworthiness_fashion_mnist", 1.0),
    ("continuity_fashion_mnist", 1.0),
))
setting("newsgroups", weighted_mean(
    ("knn_acc_newsgroups", 1.0),
    ("trustworthiness_newsgroups", 1.0),
    ("continuity_newsgroups", 1.0),
))

task(gmean("mnist", "fashion_mnist", "newsgroups"))
