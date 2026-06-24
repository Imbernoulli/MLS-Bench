# Custom CATE Estimator for MLS-Bench
#
# EDITABLE section: CATEEstimator class (the treatment effect estimator).
# FIXED sections: everything else (input loading, cross-fitting, CLI).
#
# Research question: Design a novel estimator for Conditional Average Treatment
# Effects (CATE) across explicitly synthetic observational DGP families.
#
# NOTE: The data-generating process and the ground-truth treatment effect are
# NOT part of this program. The harness pre-generates the observational inputs
# (X, T, Y) and scores your cross-fitted predictions against held-out truth in
# a separate process. Your estimator only ever sees (X, T, Y).

import os
import io
import argparse
import base64
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
# EDITABLE: Custom CATE Estimator
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
# FIXED: Input loading + cross-fitting + prediction emit
# =====================================================================

DATASETS = ("ihdp_synth", "jobs_synth", "acic_synth")


def _inputs_dir():
    """Directory holding the pre-generated (X, T, Y) inputs for this task."""
    env = os.environ.get("CATE_INPUTS_DIR")
    if env:
        return env
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), "_cate_inputs")


def load_inputs(dataset, data_seed):
    """Load the pre-generated observational inputs for (dataset, data_seed).

    Only (X, T, Y) are available — the true treatment effect is held out by the
    harness and is not present in this process.
    """
    path = os.path.join(_inputs_dir(), f"{dataset}_seed{data_seed}.npz.b64")
    with open(path, "r") as f:
        raw = base64.b64decode(f.read())
    with np.load(io.BytesIO(raw)) as d:
        return d["X"], d["T"], d["Y"]


def cross_fit_predict(X, T, Y, n_splits, seed):
    """Cross-fitted CATE prediction (same K-fold protocol as before).

    Fit on K-1 folds, predict held-out fold; aggregate to per-row tau_hat.
    A fresh deep-copy of the estimator is used per fold (cross-fitting).
    """
    import copy
    base = CATEEstimator()
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    tau_hat_all = np.zeros(len(X))
    for train_idx, test_idx in kf.split(X):
        est = copy.deepcopy(base)
        est.fit(X[train_idx], T[train_idx], Y[train_idx])
        tau_hat_all[test_idx] = est.predict(X[test_idx])
    return tau_hat_all


def main():
    parser = argparse.ArgumentParser(description="CATE Estimation Benchmark")
    parser.add_argument("--dataset", type=str, required=True,
                        choices=list(DATASETS),
                        help="Dataset to evaluate on")
    parser.add_argument("--seed", type=int, default=42,
                        help="Base random seed (selects the pre-generated inputs)")
    parser.add_argument("--n-splits", type=int, default=5,
                        help="Number of cross-validation folds")
    parser.add_argument("--n-reps", type=int, default=10,
                        help="Number of repetitions with different data seeds")
    args = parser.parse_args()

    print(f"Evaluating on {args.dataset} (seed={args.seed}, "
          f"n_splits={args.n_splits}, n_reps={args.n_reps})", flush=True)

    start = time.time()
    for rep in range(args.n_reps):
        data_seed = args.seed + rep * 1000
        X, T, Y = load_inputs(args.dataset, data_seed)
        tau_hat = cross_fit_predict(X, T, Y, args.n_splits, data_seed)

        # Emit the cross-fitted predictions for the held-out scorer. We do NOT
        # have the true tau, so we cannot (and do not) compute the metric here.
        payload = base64.b64encode(
            np.ascontiguousarray(tau_hat, dtype=np.float64).tobytes()
        ).decode("ascii")
        print(f"CATE_PRED dataset={args.dataset} seed={args.seed} rep={rep} "
              f"data_seed={data_seed} n={tau_hat.shape[0]} tau_hat={payload}",
              flush=True)

    print(f"Done {args.dataset}: emitted {args.n_reps} prediction sets in "
          f"{time.time() - start:.1f}s", flush=True)


if __name__ == "__main__":
    main()
