"""ML Ensemble Boosting Benchmark.

Train gradient-boosted ensembles of decision stumps/trees on tabular datasets
to evaluate novel sample weighting / boosting update strategies.

FIXED: Data loading, base learner (decision trees), prediction aggregation,
       evaluation loop, CLI.
EDITABLE: BoostingStrategy class — compute_sample_weights() and update_weights().

Usage:
    python custom_boosting.py --dataset breast_cancer --task classification --seed 42
    python custom_boosting.py --dataset diabetes --task regression --seed 42
"""

import argparse
import math
import os
import time
from abc import ABC, abstractmethod

import numpy as np
from sklearn.datasets import (
    fetch_california_housing,
    load_breast_cancer,
    load_diabetes,
)
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.metrics import accuracy_score, mean_squared_error


# ============================================================================
# FIXED — Data loading and preprocessing
# ============================================================================

def load_dataset(name):
    """Load a dataset by name. Returns X, y, task_type."""
    if name == "breast_cancer":
        data = load_breast_cancer()
        return data.data, data.target, "classification"
    elif name == "diabetes":
        data = load_diabetes()
        return data.data, data.target, "regression"
    elif name == "california_housing":
        data = fetch_california_housing(data_home=os.environ.get("SKLEARN_DATA_HOME"))
        return data.data, data.target, "regression"
    else:
        raise ValueError(f"Unknown dataset: {name}")


def normalize_features(X_train, X_test):
    """Standardize features to zero mean and unit variance."""
    mean = X_train.mean(axis=0)
    std = X_train.std(axis=0) + 1e-8
    return (X_train - mean) / std, (X_test - mean) / std


# ============================================================================
# FIXED — Base learner interface
# ============================================================================

class BaseLearner:
    """Wrapper around sklearn decision tree as weak learner."""

    def __init__(self, task_type, max_depth=1, random_state=None):
        self.task_type = task_type
        if task_type == "classification":
            self.tree = DecisionTreeClassifier(
                max_depth=max_depth, random_state=random_state,
            )
        else:
            self.tree = DecisionTreeRegressor(
                max_depth=max_depth, random_state=random_state,
            )

    def fit(self, X, y, sample_weight=None):
        self.tree.fit(X, y, sample_weight=sample_weight)
        return self

    def predict(self, X):
        return self.tree.predict(X)


# ============================================================================
# FIXED — Ensemble prediction and evaluation
# ============================================================================

def ensemble_predict(learners, alphas, learner_modes, X, task_type,
                     learning_rate=0.1):
    """Predict using the ensemble.

    For classification:
      - Discrete learners (AdaBoost-style): weighted majority vote with {-1,+1} coding
      - Continuous learners (gradient-based): accumulate raw scores, threshold at 0.5
    For regression:
      - First learner is the initial constant predictor
      - Subsequent learners predict residuals, scaled by alpha * learning_rate

    Args:
        learners: list of fitted BaseLearner / MeanPredictor.
        alphas: list of float learner weights.
        learner_modes: list of str, "discrete" or "continuous" per learner.
        X: np.ndarray [n_samples, n_features].
        task_type: "classification" or "regression".
        learning_rate: shrinkage for regression / gradient methods.
    """
    n_samples = X.shape[0]
    raw_scores = np.zeros(n_samples)

    for i, (learner, alpha, mode) in enumerate(zip(learners, alphas, learner_modes)):
        preds = learner.predict(X)
        if task_type == "regression":
            if i == 0:
                raw_scores += preds  # initial mean predictor
            else:
                raw_scores += alpha * learning_rate * preds
        elif mode == "discrete":
            # AdaBoost-style: convert {0,1} -> {-1,+1}
            raw_scores += alpha * (2 * preds - 1)
        else:
            # Gradient-based: accumulate continuous predictions
            raw_scores += alpha * learning_rate * preds

    if task_type == "classification":
        return (raw_scores >= 0).astype(int)
    else:
        return raw_scores


def evaluate_ensemble(learners, alphas, learner_modes, X, y, task_type,
                      learning_rate=0.1):
    """Evaluate the ensemble on given data."""
    preds = ensemble_predict(learners, alphas, learner_modes, X, task_type,
                             learning_rate)
    if task_type == "classification":
        acc = accuracy_score(y, preds)
        return {"accuracy": acc}
    else:
        rmse = np.sqrt(mean_squared_error(y, preds))
        return {"rmse": rmse}


# ============================================================================
# EDITABLE — Boosting strategy (lines 147-256)
# ============================================================================

class BoostingStrategy:
    """Sample weighting and update strategy for gradient boosting.

    This class controls how sample weights are initialized, how pseudo-targets
    (residuals or transformed targets) are computed for the next weak learner,
    how learner weights (alphas) are determined, and how sample weights are
    updated after each boosting round.

    The strategy is used by the fixed training loop (below) which:
    1. Calls init_weights() once at the start
    2. For each round t = 0..T-1:
       a. Calls compute_targets() to get pseudo-targets for fitting the learner
       b. Fits a base learner on (X, pseudo_targets, sample_weights)
       c. Calls compute_learner_weight() to get alpha_t
       d. Calls update_weights() to adjust sample weights

    Args (available via self.config set in __init__):
        n_samples: int — number of training samples
        n_features: int — number of input features
        n_rounds: int — total boosting rounds
        task_type: str — 'classification' or 'regression'
        learning_rate: float — shrinkage factor (default 0.1)
        dataset: str — dataset name

    For classification: y in {0, 1}, use signed labels y_signed = 2*y - 1
    For regression: y is continuous, use residual-based approaches
    """

    def __init__(self, config):
        """Initialize the boosting strategy.

        Args:
            config: dict with keys n_samples, n_features, n_rounds,
                    task_type, learning_rate, dataset.
        """
        self.config = config
        self.task_type = config["task_type"]
        self.n_rounds = config["n_rounds"]
        self.learning_rate = config["learning_rate"]

    def init_weights(self, n_samples):
        """Initialize sample weights.

        Args:
            n_samples: int — number of training samples.

        Returns:
            np.ndarray of shape [n_samples] — initial sample weights (should sum to 1).
        """
        return np.ones(n_samples) / n_samples

    def compute_targets(self, y, current_predictions, sample_weights, round_idx):
        """Compute pseudo-targets for the next weak learner to fit.

        This determines WHAT the weak learner tries to predict at each round.

        Args:
            y: np.ndarray [n_samples] — true labels/targets.
            current_predictions: np.ndarray [n_samples] — ensemble prediction so far
                (raw scores for classification, values for regression).
            sample_weights: np.ndarray [n_samples] — current sample weights.
            round_idx: int — current boosting round (0-indexed).

        Returns:
            np.ndarray [n_samples] — pseudo-targets to fit the weak learner on.
        """
        # Default: fit on original labels (basic boosting)
        return y

    def compute_learner_weight(self, learner, X, y, pseudo_targets,
                                sample_weights, round_idx):
        """Compute the weight (alpha) for the newly fitted learner.

        Args:
            learner: BaseLearner — the just-fitted weak learner.
            X: np.ndarray [n_samples, n_features] — training features.
            y: np.ndarray [n_samples] — true labels/targets.
            pseudo_targets: np.ndarray [n_samples] — what the learner was fit on.
            sample_weights: np.ndarray [n_samples] — current sample weights.
            round_idx: int — current boosting round.

        Returns:
            float — learner weight alpha_t. For classification, higher alpha
                means more influence in the vote. For regression, alpha scales
                the contribution (multiplied by learning_rate).
        """
        return 1.0

    def update_weights(self, sample_weights, learner, X, y, pseudo_targets,
                       alpha, round_idx):
        """Update sample weights after fitting a learner.

        This determines how the distribution over training samples shifts
        to focus on harder examples in subsequent rounds.

        Args:
            sample_weights: np.ndarray [n_samples] — current sample weights.
            learner: BaseLearner — the just-fitted weak learner.
            X: np.ndarray [n_samples, n_features] — training features.
            y: np.ndarray [n_samples] — true labels/targets.
            pseudo_targets: np.ndarray [n_samples] — what the learner was fit on.
            alpha: float — the learner's weight.
            round_idx: int — current boosting round.

        Returns:
            np.ndarray [n_samples] — updated sample weights (should sum to 1).
        """
        # Default: uniform weights (no reweighting)
        return sample_weights


# ============================================================================
# FIXED — Training loop
# ============================================================================

def train_boosting(X_train, y_train, X_test, y_test, strategy, config):
    """Train a boosted ensemble using the given strategy.

    Args:
        X_train, y_train: training data.
        X_test, y_test: test data.
        strategy: BoostingStrategy instance.
        config: dict with n_rounds, task_type, learning_rate, max_depth, seed.

    Returns:
        learners: list of fitted BaseLearner.
        alphas: list of float learner weights.
        metrics: dict of final test metrics.
    """
    n_rounds = config["n_rounds"]
    task_type = config["task_type"]
    lr = config["learning_rate"]
    max_depth = config["max_depth"]
    seed = config["seed"]

    learners = []
    alphas = []
    learner_modes = []  # "discrete" or "continuous" per learner

    # Initialize sample weights
    n_samples = X_train.shape[0]
    sample_weights = strategy.init_weights(n_samples)

    # For regression: track cumulative predictions for residual computation
    # Use a simple mean predictor as the initial model
    if task_type == "regression":
        class MeanPredictor:
            def __init__(self, mean_val):
                self._mean = mean_val
            def predict(self, X):
                return np.full(X.shape[0], self._mean)
        init_learner = MeanPredictor(y_train.mean())
        learners.append(init_learner)
        alphas.append(1.0)
        learner_modes.append("continuous")
        current_preds_train = init_learner.predict(X_train)
    else:
        current_preds_train = np.zeros(n_samples)

    for t in range(n_rounds):
        # 1. Compute pseudo-targets
        pseudo_targets = strategy.compute_targets(
            y_train, current_preds_train, sample_weights, t,
        )

        # 2. Fit weak learner
        # Use regressor if pseudo-targets are continuous (e.g. gradient boosting
        # fits residuals even for classification tasks).
        is_continuous = not np.array_equal(pseudo_targets, pseudo_targets.astype(int))
        learner_type = "regression" if is_continuous else task_type
        learner = BaseLearner(learner_type, max_depth=max_depth,
                              random_state=seed + t + 1)
        learner.fit(X_train, pseudo_targets, sample_weight=sample_weights)
        mode = "continuous" if is_continuous else "discrete"

        # 3. Compute learner weight
        alpha = strategy.compute_learner_weight(
            learner, X_train, y_train, pseudo_targets, sample_weights, t,
        )

        # 4. Update sample weights
        sample_weights = strategy.update_weights(
            sample_weights, learner, X_train, y_train, pseudo_targets, alpha, t,
        )

        # Ensure weights are valid
        sample_weights = np.clip(sample_weights, 1e-10, None)
        sample_weights = sample_weights / sample_weights.sum()

        # 5. Update cumulative predictions
        preds_t = learner.predict(X_train)
        if task_type == "classification" and mode == "discrete":
            # AdaBoost-style: discrete predictions, signed vote
            current_preds_train += alpha * (2 * preds_t - 1)
        else:
            # Gradient-based or regression: accumulate scaled predictions
            current_preds_train += alpha * lr * preds_t

        learners.append(learner)
        alphas.append(alpha)
        learner_modes.append(mode)

        # Log progress
        if (t + 1) % max(1, n_rounds // 10) == 0 or t == 0:
            test_metrics = evaluate_ensemble(
                learners, alphas, learner_modes,
                X_test, y_test, task_type, lr,
            )
            train_metrics = evaluate_ensemble(
                learners, alphas, learner_modes,
                X_train, y_train, task_type, lr,
            )
            if task_type == "classification":
                print(
                    f"TRAIN_METRICS: round={t+1}/{n_rounds} "
                    f"train_acc={train_metrics['accuracy']:.4f} "
                    f"test_acc={test_metrics['accuracy']:.4f}",
                    flush=True,
                )
            else:
                print(
                    f"TRAIN_METRICS: round={t+1}/{n_rounds} "
                    f"train_rmse={train_metrics['rmse']:.4f} "
                    f"test_rmse={test_metrics['rmse']:.4f}",
                    flush=True,
                )

    # Final evaluation
    final_metrics = evaluate_ensemble(
        learners, alphas, learner_modes, X_test, y_test, task_type, lr,
    )
    return learners, alphas, final_metrics


# ============================================================================
# FIXED — Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="ML Ensemble Boosting Benchmark")
    parser.add_argument("--dataset", type=str, required=True,
                        choices=["breast_cancer", "diabetes", "california_housing"])
    parser.add_argument("--task", type=str, required=True,
                        choices=["classification", "regression"])
    parser.add_argument("--n-rounds", type=int, default=200,
                        help="Number of boosting rounds")
    parser.add_argument("--max-depth", type=int, default=3,
                        help="Max depth of base decision trees")
    parser.add_argument("--learning-rate", type=float, default=0.1,
                        help="Shrinkage / learning rate")
    parser.add_argument("--test-size", type=float, default=0.2,
                        help="Fraction of data for testing")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=str, default=".")
    args = parser.parse_args()

    np.random.seed(args.seed)

    # Load data
    X, y, detected_task = load_dataset(args.dataset)
    task_type = args.task
    print(f"Dataset: {args.dataset} ({task_type})", flush=True)
    print(f"Samples: {X.shape[0]}, Features: {X.shape[1]}", flush=True)

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=args.seed,
    )

    # Normalize
    X_train, X_test = normalize_features(X_train, X_test)

    print(f"Train: {X_train.shape[0]}, Test: {X_test.shape[0]}", flush=True)
    print(f"Boosting rounds: {args.n_rounds}, Max depth: {args.max_depth}, "
          f"LR: {args.learning_rate}", flush=True)

    # Build strategy config
    config = {
        "n_samples": X_train.shape[0],
        "n_features": X_train.shape[1],
        "n_rounds": args.n_rounds,
        "task_type": task_type,
        "learning_rate": args.learning_rate,
        "max_depth": args.max_depth,
        "dataset": args.dataset,
        "seed": args.seed,
    }

    # Create strategy and train
    strategy = BoostingStrategy(config)
    learners, alphas, final_metrics = train_boosting(
        X_train, y_train, X_test, y_test, strategy, config,
    )

    # Report final metrics
    if task_type == "classification":
        print(f"TEST_METRICS: test_accuracy={final_metrics['accuracy']:.4f}", flush=True)
    else:
        print(f"TEST_METRICS: test_rmse={final_metrics['rmse']:.4f}", flush=True)


if __name__ == "__main__":
    main()
