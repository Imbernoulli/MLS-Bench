# Custom CATE Estimator for MLS-Bench
#
# EDITABLE section: CATEEstimator class (the treatment effect estimator).
# FIXED sections: everything else (data generation, evaluation, CLI).
#
# Research question: Design a novel estimator for Conditional Average Treatment
# Effects (CATE) across explicitly synthetic observational DGP families.

import os
import argparse
import json
import time
import warnings
from abc import ABC, abstractmethod

import numpy as np
from scipy import stats
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge, Lasso
from sklearn.ensemble import (
    RandomForestRegressor,
    RandomForestClassifier,
    GradientBoostingRegressor,
    GradientBoostingClassifier,
)
from sklearn.tree import DecisionTreeRegressor
from sklearn.neural_network import MLPRegressor, MLPClassifier
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.base import clone

warnings.filterwarnings("ignore")


# =====================================================================
# FIXED: Data Generating Processes (synthetic benchmark families)
# =====================================================================

def generate_ihdp(n=747, p=25, seed=42):
    """Task-local synthetic IHDP-inspired DGP.

    This is not the official IHDP benchmark. It preserves the small-sample,
    mixed-feature, nonlinear-effect flavor of IHDP-style CATE evaluations while
    using fully synthetic covariates, treatment assignment, and outcomes.

    Features strong confounding: propensity depends on the same variables
    that drive heterogeneous treatment effects, with nonlinear interactions.

    Returns:
        X: (n, p) covariate matrix
        T: (n,) binary treatment indicator
        Y: (n,) observed outcomes
        tau: (n,) true individual treatment effects (CATE)
        ate: scalar true average treatment effect
    """
    rng = np.random.RandomState(seed)

    # Covariates: mix of continuous and binary
    X_cont = rng.randn(n, 6)
    X_bin = rng.binomial(1, 0.5, size=(n, p - 6)).astype(float)
    X = np.hstack([X_cont, X_bin])

    # Nonlinear propensity score with interactions (strong confounding)
    logit_e = (
        0.5 * X[:, 0]
        - 0.3 * X[:, 1]
        + 0.2 * X[:, 2]
        + 0.15 * X[:, 0] * X[:, 1]       # interaction
        - 0.2 * X[:, 2] ** 2              # quadratic
        + 0.25 * X[:, 3] * X[:, 5]        # confounders shared with tau
        + 0.1 * np.sin(X[:, 4] * np.pi)   # nonlinear
    )
    e = 1.0 / (1.0 + np.exp(-logit_e))
    e = np.clip(e, 0.05, 0.95)
    T = rng.binomial(1, e)

    # Response surfaces (complex nonlinear)
    mu0 = (
        np.exp(0.8 * X[:, 0] + 0.5 * X[:, 1])
        + X[:, 2] * X[:, 3]
        + 0.5 * X[:, 4]
        + 0.3 * X[:, 0] * X[:, 2]
        + 0.2 * np.cos(X[:, 5] * np.pi)
        + rng.randn(n) * 0.5
    )
    # Heterogeneous treatment effect with confounder-dependent terms
    tau = (
        1.0
        + 0.5 * X[:, 0]
        + 0.3 * X[:, 1] ** 2
        - 0.4 * X[:, 2] * X[:, 5]         # interaction with confounder
        + 0.5 * np.sin(X[:, 3] * np.pi)
        + 0.3 * np.maximum(X[:, 4], 0)     # ReLU-like
        - 0.2 * X[:, 0] * X[:, 3]          # cross-interaction
        + 0.15 * X[:, 6]                   # binary covariate effect
    )
    mu1 = mu0 + tau + rng.randn(n) * 0.5

    Y = T * mu1 + (1 - T) * mu0
    ate = tau.mean()

    return X, T, Y, tau, ate


def generate_jobs(n=2000, p=10, seed=42):
    """Task-local synthetic Jobs/LaLonde-inspired DGP.

    This is not the official NSW/LaLonde or Jobs benchmark. It simulates a job
    training program with observational assignment, earnings-like outcomes, and
    heterogeneous effects over age, education, earnings, and demographics.

    Features nonlinear confounding and complex treatment effect heterogeneity
    with interactions between confounders and effect modifiers.

    Returns:
        X: (n, p) covariate matrix
        T: (n,) binary treatment indicator
        Y: (n,) observed outcomes (earnings)
        tau: (n,) true individual treatment effects
        ate: scalar true average treatment effect
    """
    rng = np.random.RandomState(seed)

    # Covariates simulating demographic features
    age = rng.uniform(18, 55, n)
    education = rng.uniform(8, 18, n)
    prior_earnings = np.maximum(0, rng.normal(10000, 5000, n))
    married = rng.binomial(1, 0.4, n).astype(float)
    black = rng.binomial(1, 0.3, n).astype(float)
    hispanic = rng.binomial(1, 0.15, n).astype(float)

    # Additional covariates
    extra = rng.randn(n, p - 6)
    X = np.column_stack([age, education, prior_earnings, married, black, hispanic, extra])

    # Normalize for stability
    X_scaled = (X - X.mean(0)) / (X.std(0) + 1e-8)

    # Nonlinear propensity with interactions (strong confounding)
    logit_e = (
        -0.3 * X_scaled[:, 0]
        - 0.2 * X_scaled[:, 1]
        - 0.4 * X_scaled[:, 2]
        + 0.1 * X_scaled[:, 3]
        + 0.25 * X_scaled[:, 0] * X_scaled[:, 1]   # age-education interaction
        - 0.15 * X_scaled[:, 2] ** 2                # quadratic earnings effect
        + 0.2 * X_scaled[:, 1] * X_scaled[:, 3]     # education-married interaction
    )
    e = 1.0 / (1.0 + np.exp(-logit_e))
    e = np.clip(e, 0.05, 0.95)
    T = rng.binomial(1, e)

    # Base outcome (earnings) with nonlinearities
    mu0 = (
        5000
        + 200 * X_scaled[:, 0]
        + 500 * X_scaled[:, 1]
        + 0.3 * prior_earnings
        + 1000 * married
        + 300 * X_scaled[:, 0] * X_scaled[:, 1]     # age-education interaction
        + 200 * np.maximum(X_scaled[:, 2], 0)        # ReLU on earnings
        + 150 * X_scaled[:, 4] * X_scaled[:, 5]      # race interaction
        + rng.randn(n) * 800
    )

    # Complex heterogeneous treatment effect
    tau = (
        1500
        + 300 * X_scaled[:, 1]                        # more education -> bigger effect
        - 200 * X_scaled[:, 0]                        # younger -> bigger effect
        + 250 * np.abs(X_scaled[:, 2])                # nonlinear prior earnings
        + 100 * X_scaled[:, 3]
        + 400 * np.sin(X_scaled[:, 0] * np.pi / 2)   # periodic age effect
        - 200 * X_scaled[:, 1] * X_scaled[:, 2]       # education-earnings interaction
        + 300 * np.maximum(X_scaled[:, 6], 0)          # ReLU on extra covariate
        + 150 * X_scaled[:, 0] * X_scaled[:, 3]       # age-married interaction
    )

    mu1 = mu0 + tau + rng.randn(n) * 500
    Y = T * mu1 + (1 - T) * mu0
    ate = tau.mean()

    return X, T, Y, tau, ate


def generate_acic(n=4000, p=50, seed=42):
    """Task-local synthetic ACIC-inspired DGP.

    This is not an official ACIC competition scenario. It uses synthetic
    high-dimensional correlated covariates with complex nonlinear response
    surfaces and strong confounding to test robustness to misspecification.

    Returns:
        X: (n, p) covariate matrix
        T: (n,) binary treatment indicator
        Y: (n,) observed outcomes
        tau: (n,) true individual treatment effects
        ate: scalar true average treatment effect
    """
    rng = np.random.RandomState(seed)

    # High-dimensional covariates with correlations
    mean = np.zeros(p)
    # Block-diagonal correlation structure
    cov = np.eye(p)
    for i in range(0, p - 1, 2):
        cov[i, i + 1] = 0.3
        cov[i + 1, i] = 0.3
    X = rng.multivariate_normal(mean, cov, n)

    # Complex propensity model (strong confounding)
    logit_e = (
        0.4 * X[:, 0]
        + 0.3 * X[:, 1]
        - 0.2 * X[:, 2]
        + 0.15 * X[:, 0] * X[:, 1]
        - 0.1 * X[:, 3] ** 2
        + 0.05 * np.sum(X[:, 4:10], axis=1)
    )
    e = 1.0 / (1.0 + np.exp(-logit_e))
    e = np.clip(e, 0.05, 0.95)  # Overlap enforcement
    T = rng.binomial(1, e)

    # Complex response surface (nonlinear, interactions)
    mu0 = (
        2.0 * np.sin(X[:, 0] * np.pi)
        + X[:, 1] ** 2
        + 0.5 * X[:, 2] * X[:, 3]
        - 1.5 * np.abs(X[:, 4])
        + 0.3 * np.sum(X[:, 5:15], axis=1)
        + rng.randn(n) * 0.5
    )

    # Complex heterogeneous treatment effect
    tau = (
        0.8
        + 0.6 * X[:, 0]
        - 0.3 * X[:, 1] ** 2
        + 0.4 * np.maximum(X[:, 2], 0)
        + 0.2 * X[:, 3] * X[:, 4]
        - 0.15 * np.abs(X[:, 5])
        + 0.1 * np.cos(X[:, 6] * np.pi)
    )

    mu1 = mu0 + tau + rng.randn(n) * 0.3
    Y = T * mu1 + (1 - T) * mu0
    ate = tau.mean()

    return X, T, Y, tau, ate


# =====================================================================
# FIXED: Base class for CATE estimators
# =====================================================================

class BaseCATEEstimator(ABC):
    """Abstract base class for CATE estimators.

    All estimators must implement:
        fit(X, T, Y) -> self
        predict(X) -> tau_hat array of shape (n,)
    """

    @abstractmethod
    def fit(self, X, T, Y):
        """Fit the estimator on observational data.

        Args:
            X: (n, p) covariate matrix (numpy array)
            T: (n,) binary treatment indicator (0 or 1)
            Y: (n,) observed outcomes (continuous)

        Returns:
            self
        """
        pass

    @abstractmethod
    def predict(self, X):
        """Predict CATE for given covariates.

        Args:
            X: (n, p) covariate matrix

        Returns:
            tau_hat: (n,) array of estimated treatment effects
        """
        pass


# =====================================================================
# FIXED: Evaluation utilities
# =====================================================================

def compute_pehe(tau_true, tau_hat):
    """Precision in Estimation of Heterogeneous Effects (lower is better).

    PEHE = sqrt(mean((tau_hat - tau_true)^2))
    """
    return np.sqrt(np.mean((tau_hat - tau_true) ** 2))


def compute_ate_error(ate_true, tau_hat):
    """Absolute error in ATE estimation (lower is better).

    ATE_error = |mean(tau_hat) - ATE_true|
    """
    return np.abs(np.mean(tau_hat) - ate_true)


def evaluate_estimator(estimator, X, T, Y, tau, ate, n_splits=5, seed=42):
    """Evaluate CATE estimator using cross-fitting.

    Performs K-fold cross-validation: fit on K-1 folds, predict on held-out fold.
    Aggregates PEHE and ATE error across all held-out predictions.

    Returns:
        dict with PEHE and ATE_error metrics
    """
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    tau_hat_all = np.zeros(len(X))

    for train_idx, test_idx in kf.split(X):
        est = clone_estimator(estimator)
        est.fit(X[train_idx], T[train_idx], Y[train_idx])
        tau_hat_all[test_idx] = est.predict(X[test_idx])

    pehe = compute_pehe(tau, tau_hat_all)
    ate_err = compute_ate_error(ate, tau_hat_all)

    return {"PEHE": pehe, "ATE_error": ate_err}


def clone_estimator(estimator):
    """Create a fresh copy of a CATE estimator."""
    import copy
    return copy.deepcopy(estimator)


# =====================================================================
# EDITABLE: Custom CATE Estimator (lines 344-416)
# =====================================================================

class CATEEstimator(BaseCATEEstimator):
    """Custom CATE (Conditional Average Treatment Effect) estimator.

    Design a novel estimator for heterogeneous treatment effects from
    observational data. Your estimator receives covariates X, binary
    treatment T, and outcomes Y, and must estimate tau(x) = E[Y(1)-Y(0)|X=x].

    Key challenges:
    - Confounding: treatment assignment depends on covariates
    - Heterogeneity: treatment effects vary across individuals
    - Model misspecification: response surfaces may be nonlinear
    - Finite-sample performance: must work well with limited data

    Approaches to consider:
    - Meta-learners (S/T/X/R/DR-Learner frameworks)
    - Propensity score methods (weighting, matching, doubly robust)
    - Tree-based methods (causal forests, Bayesian additive regression trees)
    - Representation learning for treatment effects
    - Kernel methods or local regression for CATE
    - Ensemble methods combining multiple estimators

    Available imports (in FIXED section above):
        numpy, scipy.stats, sklearn (all submodules)

    Interface contract:
        fit(X, T, Y) -> self
        predict(X) -> tau_hat of shape (n,)
    """

    def __init__(self):
        """Initialize the CATE estimator.

        TODO: Set up any models, hyperparameters, or data structures needed.
        """
        pass

    def fit(self, X, T, Y):
        """Fit the estimator on observational data.

        Args:
            X: (n, p) numpy array of covariates
            T: (n,) numpy array of binary treatment indicators (0 or 1)
            Y: (n,) numpy array of observed outcomes

        Returns:
            self

        TODO: Implement your CATE estimation algorithm.
        The default implementation is a simple S-Learner placeholder.
        """
        # Placeholder: simple S-Learner (augmented features)
        n, p = X.shape
        XT = np.column_stack([X, T.reshape(-1, 1)])
        self._model = Ridge(alpha=1.0)
        self._model.fit(XT, Y)
        return self

    def predict(self, X):
        """Predict CATE for given covariates.

        Args:
            X: (n, p) numpy array of covariates

        Returns:
            tau_hat: (n,) numpy array of estimated treatment effects

        TODO: Implement prediction of individual treatment effects.
        """
        n = X.shape[0]
        X1 = np.column_stack([X, np.ones((n, 1))])
        X0 = np.column_stack([X, np.zeros((n, 1))])
        return self._model.predict(X1) - self._model.predict(X0)


# =====================================================================
# FIXED: Main evaluation loop
# =====================================================================

DATASETS = {
    "ihdp_synth": generate_ihdp,
    "jobs_synth": generate_jobs,
    "acic_synth": generate_acic,
}


def main():
    parser = argparse.ArgumentParser(description="CATE Estimation Benchmark")
    parser.add_argument("--dataset", type=str, required=True,
                        choices=list(DATASETS.keys()),
                        help="Dataset to evaluate on")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for data generation and evaluation")
    parser.add_argument("--n-splits", type=int, default=5,
                        help="Number of cross-validation folds")
    parser.add_argument("--n-reps", type=int, default=10,
                        help="Number of repetitions with different data seeds")
    args = parser.parse_args()

    print(f"Evaluating on {args.dataset} (seed={args.seed}, "
          f"n_splits={args.n_splits}, n_reps={args.n_reps})", flush=True)

    pehe_values = []
    ate_err_values = []

    for rep in range(args.n_reps):
        data_seed = args.seed + rep * 1000
        X, T, Y, tau, ate = DATASETS[args.dataset](seed=data_seed)

        estimator = CATEEstimator()
        metrics = evaluate_estimator(
            estimator, X, T, Y, tau, ate,
            n_splits=args.n_splits, seed=data_seed,
        )

        pehe_values.append(metrics["PEHE"])
        ate_err_values.append(metrics["ATE_error"])

        print(f"TRAIN_METRICS rep={rep} PEHE={metrics['PEHE']:.6f} "
              f"ATE_error={metrics['ATE_error']:.6f}", flush=True)

    # Aggregate across repetitions
    mean_pehe = np.mean(pehe_values)
    std_pehe = np.std(pehe_values)
    mean_ate_err = np.mean(ate_err_values)
    std_ate_err = np.std(ate_err_values)

    print(f"\n=== Results on {args.dataset} ===", flush=True)
    print(f"PEHE: {mean_pehe:.6f} +/- {std_pehe:.6f}", flush=True)
    print(f"ATE_error: {mean_ate_err:.6f} +/- {std_ate_err:.6f}", flush=True)

    # Final metrics for parser
    print(f"TEST_METRICS PEHE={mean_pehe:.6f} ATE_error={mean_ate_err:.6f}", flush=True)


if __name__ == "__main__":
    main()
