"""Score spec for ml-calibration.

Four settings (rf-mnist, mlp-fashion_mnist, svm-breast_cancer, gbm-madelon).
Each with ECE, Brier, NLL — all lower-is-better with theoretical bound 0.
Normalization uses dynamic leaderboard anchors: worst baseline = 0-point floor,
best baseline = 50-point anchor.
"""
from mlsbench.scoring.dsl import *

# --- rf-mnist ---
term("ece_rf_mnist",
    col("ECE_rf-mnist").lower().id()
    .bounded_power(bound=0.0))
term("brier_rf_mnist",
    col("Brier_rf-mnist").lower().id()
    .bounded_power(bound=0.0))
term("nll_rf_mnist",
    col("NLL_rf-mnist").lower().id()
    .bounded_power(bound=0.0))

setting("rf-mnist", weighted_mean(
    ("ece_rf_mnist", 1.0),
    ("brier_rf_mnist", 1.0),
    ("nll_rf_mnist", 1.0),
))

# --- mlp-fashion_mnist ---
term("ece_mlp_fmnist",
    col("ECE_mlp-fashion_mnist").lower().id()
    .bounded_power(bound=0.0))
term("brier_mlp_fmnist",
    col("Brier_mlp-fashion_mnist").lower().id()
    .bounded_power(bound=0.0))
term("nll_mlp_fmnist",
    col("NLL_mlp-fashion_mnist").lower().id()
    .bounded_power(bound=0.0))

setting("mlp-fashion_mnist", weighted_mean(
    ("ece_mlp_fmnist", 1.0),
    ("brier_mlp_fmnist", 1.0),
    ("nll_mlp_fmnist", 1.0),
))

# --- svm-breast_cancer ---
term("ece_svm_bc",
    col("ECE_svm-breast_cancer").lower().id()
    .bounded_power(bound=0.0))
term("brier_svm_bc",
    col("Brier_svm-breast_cancer").lower().id()
    .bounded_power(bound=0.0))
term("nll_svm_bc",
    col("NLL_svm-breast_cancer").lower().id()
    .bounded_power(bound=0.0))

setting("svm-breast_cancer", weighted_mean(
    ("ece_svm_bc", 1.0),
    ("brier_svm_bc", 1.0),
    ("nll_svm_bc", 1.0),
))

# --- gbm-madelon ---
term("ece_gbm_madelon",
    col("ECE_gbm-madelon").lower().id()
    .bounded_power(bound=0.0))
term("brier_gbm_madelon",
    col("Brier_gbm-madelon").lower().id()
    .bounded_power(bound=0.0))
term("nll_gbm_madelon",
    col("NLL_gbm-madelon").lower().id()
    .bounded_power(bound=0.0))

setting("gbm-madelon", weighted_mean(
    ("ece_gbm_madelon", 1.0),
    ("brier_gbm_madelon", 1.0),
    ("nll_gbm_madelon", 1.0),
))

# Task: geometric mean across settings
task(gmean("rf-mnist", "mlp-fashion_mnist", "svm-breast_cancer", "gbm-madelon"))
