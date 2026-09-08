"""Score spec for ml-active-learning."""
from mlsbench.scoring.dsl import *

# accuracy is on 0-1 scale based on leaderboard values (0.8318, 0.9313, 0.8087)
# auc is on 0-1 scale
term("accuracy_letter",
    col("accuracy_letter").higher().id()
    .bounded_power(bound=1.0))

term("auc_letter",
    col("auc_letter").higher().id()
    .bounded_power(bound=1.0))

term("accuracy_spambase",
    col("accuracy_spambase").higher().id()
    .bounded_power(bound=1.0))

term("auc_spambase",
    col("auc_spambase").higher().id()
    .bounded_power(bound=1.0))

term("accuracy_splice",
    col("accuracy_splice").higher().id()
    .bounded_power(bound=1.0))

term("auc_splice",
    col("auc_splice").higher().id()
    .bounded_power(bound=1.0))

setting("letter", weighted_mean(("accuracy_letter", 1.0), ("auc_letter", 1.0)))
setting("spambase", weighted_mean(("accuracy_spambase", 1.0), ("auc_spambase", 1.0)))
setting("splice", weighted_mean(("accuracy_splice", 1.0), ("auc_splice", 1.0)))

task(gmean("letter", "spambase", "splice"))
