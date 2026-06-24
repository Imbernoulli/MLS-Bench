"""ML Ensemble Boosting Benchmark.

Train gradient-boosted ensembles of shallow decision trees on standardized
tabular data to evaluate novel sample weighting / boosting update strategies.

EDITABLE: BoostingStrategy class -- the agent's boosting strategy.
FIXED: input loading + base learner + ensemble accumulation + prediction emit.
       The dataset identity, the train/test split, the test labels, and the
       metric live in a host-only module the agent's process cannot import;
       this program loads a pre-generated standardized (X_train, y_train, X_test)
       triple, builds the boosting ensemble with the agent's strategy on the
       training split, and emits the ensemble's test predictions. The host-side
       parser regenerates the labels and scores the same metric. Inputs are
       pre-standardized, exactly as before; the split is identical.
"""

import io
import os
import base64
import warnings
from abc import ABC, abstractmethod

import numpy as np
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

warnings.filterwarnings("ignore")


# ============================================================================
# FIXED -- Base learner interface (do not modify)
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
# FIXED -- Ensemble prediction (do not modify)
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


# ============================================================================
# EDITABLE -- Boosting strategy (lines 105 to 212)
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
        n_samples: int -- number of training samples
        n_features: int -- number of input features
        n_rounds: int -- total boosting rounds
        task_type: str -- 'classification' or 'regression'
        learning_rate: float -- shrinkage factor (default 0.1)

    For classification: y in {0, 1}, use signed labels y_signed = 2*y - 1
    For regression: y is continuous, use residual-based approaches
    """

    def __init__(self, config):
        """Initialize the boosting strategy.

        Args:
            config: dict with keys n_samples, n_features, n_rounds,
                    task_type, learning_rate.
        """
        self.config = config
        self.task_type = config["task_type"]
        self.n_rounds = config["n_rounds"]
        self.learning_rate = config["learning_rate"]

    def init_weights(self, n_samples):
        """Initialize sample weights.

        Args:
            n_samples: int -- number of training samples.

        Returns:
            np.ndarray of shape [n_samples] -- initial sample weights (should sum to 1).
        """
        return np.ones(n_samples) / n_samples

    def compute_targets(self, y, current_predictions, sample_weights, round_idx):
        """Compute pseudo-targets for the next weak learner to fit.

        This determines WHAT the weak learner tries to predict at each round.

        Args:
            y: np.ndarray [n_samples] -- true labels/targets.
            current_predictions: np.ndarray [n_samples] -- ensemble prediction so far
                (raw scores for classification, values for regression).
            sample_weights: np.ndarray [n_samples] -- current sample weights.
            round_idx: int -- current boosting round (0-indexed).

        Returns:
            np.ndarray [n_samples] -- pseudo-targets to fit the weak learner on.
        """
        # Default: fit on original labels (basic boosting)
        return y

    def compute_learner_weight(self, learner, X, y, pseudo_targets,
                                sample_weights, round_idx):
        """Compute the weight (alpha) for the newly fitted learner.

        Args:
            learner: BaseLearner -- the just-fitted weak learner.
            X: np.ndarray [n_samples, n_features] -- training features.
            y: np.ndarray [n_samples] -- true labels/targets.
            pseudo_targets: np.ndarray [n_samples] -- what the learner was fit on.
            sample_weights: np.ndarray [n_samples] -- current sample weights.
            round_idx: int -- current boosting round.

        Returns:
            float -- learner weight alpha_t. For classification, higher alpha
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
            sample_weights: np.ndarray [n_samples] -- current sample weights.
            learner: BaseLearner -- the just-fitted weak learner.
            X: np.ndarray [n_samples, n_features] -- training features.
            y: np.ndarray [n_samples] -- true labels/targets.
            pseudo_targets: np.ndarray [n_samples] -- what the learner was fit on.
            alpha: float -- the learner's weight.
            round_idx: int -- current boosting round.

        Returns:
            np.ndarray [n_samples] -- updated sample weights (should sum to 1).
        """
        # Default: uniform weights (no reweighting)
        return sample_weights


# ============================================================================
# FIXED -- Training loop + input loading + prediction emit (do not modify below)
# ============================================================================
# The dataset generator (incl. identity), the train/test split, the test labels,
# and the metric live in a host-only module the agent's process cannot import.
# This program loads the pre-generated standardized (X_train, y_train, X_test)
# triple, builds the boosting ensemble using the agent's strategy on the training
# split, predicts on the held-out test split, and emits those predictions. The
# host-side parser regenerates the truth and scores it.

def train_boosting(X_train, y_train, strategy, config):
    """Train a boosted ensemble using the given strategy on the training split.

    Args:
        X_train, y_train: training data.
        strategy: BoostingStrategy instance.
        config: dict with n_rounds, task_type, learning_rate, max_depth, seed.

    Returns:
        learners: list of fitted BaseLearner.
        alphas: list of float learner weights.
        learner_modes: list of "discrete"/"continuous" per learner.
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

        # Log training progress (train-split only; test split is held out)
        if (t + 1) % max(1, n_rounds // 10) == 0 or t == 0:
            train_preds = ensemble_predict(
                learners, alphas, learner_modes, X_train, task_type, lr,
            )
            if task_type == "classification":
                train_acc = float(np.mean(train_preds == y_train))
                print(
                    f"TRAIN_METRICS: round={t+1}/{n_rounds} "
                    f"train_acc={train_acc:.4f}",
                    flush=True,
                )
            else:
                train_rmse = float(np.sqrt(np.mean((train_preds - y_train) ** 2)))
                print(
                    f"TRAIN_METRICS: round={t+1}/{n_rounds} "
                    f"train_rmse={train_rmse:.4f}",
                    flush=True,
                )

    return learners, alphas, learner_modes


def _inputs_dir():
    d = os.environ.get("BOOST_INPUTS_DIR")
    if d:
        return d
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), "_boost_inputs")


def _load_input(env_name, seed):
    path = os.path.join(_inputs_dir(), f"{env_name}_seed{seed}.npz.b64")
    with open(path, "r") as f:
        raw = base64.b64decode(f.read())
    d = np.load(io.BytesIO(raw))
    return (d["X_train"], d["y_train"], d["X_test"],
            str(d["task_type"]), int(d["n_rounds"]),
            int(d["max_depth"]), float(d["learning_rate"]))


def main():
    env = os.environ.get("ENV", "")
    if not env:
        raise SystemExit("ENV not set")
    seed = int(os.environ.get("SEED", "42"))
    print(f"=== Boosting benchmark: {env} (seed={seed}) ===", flush=True)

    X_train, y_train, X_test, task_type, n_rounds, max_depth, lr = _load_input(
        env, seed,
    )
    np.random.seed(seed)
    print(f"Input: train={X_train.shape}, test={X_test.shape}, "
          f"task={task_type}", flush=True)
    print(f"Boosting rounds: {n_rounds}, Max depth: {max_depth}, "
          f"LR: {lr}", flush=True)

    # Scrub the dataset identity and split seed from the environment before the
    # strategy is constructed, so the editable strategy cannot read ENV/SEED at
    # run time, reconstruct the public loader + split, and recover the held-out
    # test labels.
    for _k in ("ENV", "SEED", "BOOST_INPUTS_DIR"):
        os.environ.pop(_k, None)

    # config for the strategy: NO dataset identity, NO seed (cannot replay split)
    config = {
        "n_samples": X_train.shape[0],
        "n_features": X_train.shape[1],
        "n_rounds": n_rounds,
        "task_type": task_type,
        "learning_rate": lr,
    }
    # runner config carries seed/max_depth for reproducible tree fitting
    run_config = dict(config)
    run_config["max_depth"] = max_depth
    run_config["seed"] = seed

    strategy = BoostingStrategy(config)
    learners, alphas, learner_modes = train_boosting(
        X_train, y_train, strategy, run_config,
    )

    # Predict on the held-out test split and emit predictions for host scoring.
    test_preds = ensemble_predict(
        learners, alphas, learner_modes, X_test, task_type, lr,
    )
    test_preds = np.asarray(test_preds, dtype=np.float64).ravel()
    payload = base64.b64encode(
        np.ascontiguousarray(test_preds, dtype=np.float64).tobytes()
    ).decode("ascii")
    print(
        f"BOOST_PRED env={env} seed={seed} n={test_preds.shape[0]} "
        f"preds={payload}",
        flush=True,
    )
    print("Done.", flush=True)


if __name__ == "__main__":
    main()
