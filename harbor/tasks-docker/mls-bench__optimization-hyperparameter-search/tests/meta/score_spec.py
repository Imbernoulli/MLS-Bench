"""Score spec for optimization-hyperparameter-search."""
from mlsbench.scoring.dsl import *

# best_val_score: higher is better (best validation score found)
# convergence_auc: higher is better. The loop can overshoot the budget by one
# fidelity unit, so AUC bounds are (budget + 1) / budget.
# total_evals: informational count — dropped
# refs from best baseline means

term("best_val_score_xgboost",
    col("best_val_score_xgboost").higher().id()
    .sigmoid())

term("convergence_auc_xgboost",
    col("convergence_auc_xgboost").higher().id()
    .bounded_power(bound=1.02))

term("best_val_score_svm",
    col("best_val_score_svm").higher().id()
    .sigmoid())

term("convergence_auc_svm",
    col("convergence_auc_svm").higher().id()
    .bounded_power(bound=1.025))

term("best_val_score_nn",
    col("best_val_score_nn").higher().id()
    .sigmoid())

term("convergence_auc_nn",
    col("convergence_auc_nn").higher().id()
    .bounded_power(bound=1.025))

setting("xgboost", weighted_mean(
    ("best_val_score_xgboost", 1.0),
    ("convergence_auc_xgboost", 1.0),
))
setting("svm", weighted_mean(
    ("best_val_score_svm", 1.0),
    ("convergence_auc_svm", 1.0),
))
setting("nn", weighted_mean(
    ("best_val_score_nn", 1.0),
    ("convergence_auc_nn", 1.0),
))

task(gmean("xgboost", "svm", "nn"))
