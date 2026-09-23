"""Score spec for optimization-hyperparameter-search."""
from mlsbench.scoring.dsl import *

# best_val_score: higher is better (best validation score found)
# convergence_auc: higher is better. It is the mean, over the cost axis
# [0, budget], of the incumbent score normalized with FIXED per-benchmark
# references (constant predictor -> 0, perfect predictor -> 1; cost past the
# budget is not integrated), so it lies in [0, 1]. The baselines sit far
# below that bound (the best nn config has R^2 ~ 0.49), so bounded_power
# would put r(ref) < 0.05 and fall back to sigmoid anyway; sigmoid is used
# explicitly (worst baseline -> 0, best baseline -> 0.5).
# best_val_score_svm is reported but not scored: every baseline reaches the
# same 5-fold CV accuracy (0.978932), so its anchors cannot discriminate
# (worst == best -> a step at the anchor). The svm setting is scored on
# convergence_auc_svm, which still rewards a higher final accuracy.
# total_evals: informational count — dropped
# refs from best baseline means

term("best_val_score_xgboost",
    col("best_val_score_xgboost").higher().id()
    .sigmoid())

term("convergence_auc_xgboost",
    col("convergence_auc_xgboost").higher().id()
    .sigmoid())

term("convergence_auc_svm",
    col("convergence_auc_svm").higher().id()
    .sigmoid())

term("best_val_score_nn",
    col("best_val_score_nn").higher().id()
    .sigmoid())

term("convergence_auc_nn",
    col("convergence_auc_nn").higher().id()
    .sigmoid())

setting("xgboost", weighted_mean(
    ("best_val_score_xgboost", 1.0),
    ("convergence_auc_xgboost", 1.0),
))
setting("svm", weighted_mean(
    ("convergence_auc_svm", 1.0),
))
setting("nn", weighted_mean(
    ("best_val_score_nn", 1.0),
    ("convergence_auc_nn", 1.0),
))

task(gmean("xgboost", "svm", "nn"))
