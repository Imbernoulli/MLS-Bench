"""Score spec for ml-calibration.

Four settings (rf-mnist, mlp-fashion_mnist, svm-breast_cancer, gbm-madelon).
Each with ECE, Brier, NLL — all lower-is-better with theoretical bound 0
(gbm-madelon scores ECE and NLL only; see the note at its Brier term).
Normalization uses dynamic leaderboard anchors: worst baseline = 0-point floor,
best baseline = 50-point anchor.

Constraint per setting: a ceiling on the Brier score. ECE alone is trivially
gameable: predicting the class prior (or uniform) for every sample puts all
mass in one bin whose confidence equals its accuracy, so ECE ~ 0 while the
classifier's discrimination is thrown away. Measured with the fixed harness,
the constant-prior calibrator gets ECE 0.000 / 0.000 / 0.000 / 0.009 and
out-scored isotonic and Platt (0.310 vs 0.216 / 0.161). Post-hoc calibration
must not destroy what the classifier learned, so each ceiling is the
uncalibrated classifier's test Brier (the fixed harness's before_calibration
line) plus ~0.02. Every baseline passes at penalty 1.0; the constant prior
(Brier 0.90 / 0.90 / 0.25 / 0.23) is damped to ~0 (sharpness 50: 0.01 over the
ceiling costs x0.61, 0.05 costs x0.08).

    setting             uncalibrated  worst baseline  constant prior  ceiling
    rf-mnist            0.1602        0.0788 (iso)    0.8996          0.18
    mlp-fashion_mnist   0.1993        0.1877 (temp)   0.9000          0.22
    svm-breast_cancer   0.0275        0.0333 (iso)    0.2328          0.05
    gbm-madelon         0.1514        0.1521 (platt)  0.2500          0.17
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

term("brier_cap_rf_mnist",
    penalty_upper(col("Brier_rf-mnist").lower(), target=0.18, sharpness=50.0))

setting("rf-mnist", weighted_mean(
    ("ece_rf_mnist", 1.0),
    ("brier_rf_mnist", 1.0),
    ("nll_rf_mnist", 1.0),
), constraints=["brier_cap_rf_mnist"])

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

term("brier_cap_mlp_fmnist",
    penalty_upper(col("Brier_mlp-fashion_mnist").lower(), target=0.22, sharpness=50.0))

setting("mlp-fashion_mnist", weighted_mean(
    ("ece_mlp_fmnist", 1.0),
    ("brier_mlp_fmnist", 1.0),
    ("nll_mlp_fmnist", 1.0),
), constraints=["brier_cap_mlp_fmnist"])

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

term("brier_cap_svm_bc",
    penalty_upper(col("Brier_svm-breast_cancer").lower(), target=0.05, sharpness=50.0))

setting("svm-breast_cancer", weighted_mean(
    ("ece_svm_bc", 1.0),
    ("brier_svm_bc", 1.0),
    ("nll_svm_bc", 1.0),
), constraints=["brier_cap_svm_bc"])

# --- gbm-madelon ---
term("ece_gbm_madelon",
    col("ECE_gbm-madelon").lower().id()
    .bounded_power(bound=0.0))
# No Brier objective on gbm-madelon. A post-hoc map of a binary classifier's
# scores barely moves its Brier, and the three baselines sit within 1.6e-4 of
# each other (platt 0.152134, temperature 0.151990, isotonic 0.151974). That is
# below the method-to-method noise: over seeds 43-46 the paired gap between
# the same three baselines is 1e-4 to 3e-3 and changes sign. With so tight a
# spread the term fell to the sigmoid fallback at scale 1.5e-4, so 0.0005 of
# Brier swung it from ~0 to ~1 -- a coin flip on noise. ECE and NLL still
# separate the methods here, and the Brier ceiling below still applies.
term("nll_gbm_madelon",
    col("NLL_gbm-madelon").lower().id()
    .bounded_power(bound=0.0))

term("brier_cap_gbm_madelon",
    penalty_upper(col("Brier_gbm-madelon").lower(), target=0.17, sharpness=50.0))

setting("gbm-madelon", weighted_mean(
    ("ece_gbm_madelon", 1.0),
    ("nll_gbm_madelon", 1.0),
), constraints=["brier_cap_gbm_madelon"])

# Task: geometric mean across settings
task(gmean("rf-mnist", "mlp-fashion_mnist", "svm-breast_cancer", "gbm-madelon"))
