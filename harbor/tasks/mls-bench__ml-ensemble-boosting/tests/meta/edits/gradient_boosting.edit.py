"""Gradient Boosting (Friedman, 2001) baseline for ml-ensemble-boosting.

Standard gradient boosting: fit weak learners to negative gradients of the
loss function (pseudo-residuals). Uses squared error loss for regression
and log-loss (logistic) for classification.

Reference: Friedman, J.H. "Greedy Function Approximation: A Gradient Boosting
Machine." Annals of Statistics, 2001.
"""

_FILE = "scikit-learn/custom_boosting.py"

_CONTENT = """\
class BoostingStrategy:
    \"\"\"Gradient Boosting: negative gradient (pseudo-residual) fitting.\"\"\"

    def __init__(self, config):
        self.config = config
        self.task_type = config["task_type"]
        self.n_rounds = config["n_rounds"]
        self.learning_rate = config["learning_rate"]
        # Track raw scores for logistic gradient computation
        self._raw_scores = None

    def init_weights(self, n_samples):
        # Gradient boosting uses uniform weights (no reweighting);
        # the key insight is fitting to pseudo-residuals instead.
        self._raw_scores = np.zeros(n_samples)
        return np.ones(n_samples) / n_samples

    def _sigmoid(self, x):
        return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))

    def compute_targets(self, y, current_predictions, sample_weights, round_idx):
        if self.task_type == "regression":
            # Negative gradient of squared error = residuals
            return y - current_predictions
        else:
            # Negative gradient of log-loss (logistic)
            # For log-loss: -dL/dF = y - sigmoid(F)
            probs = self._sigmoid(self._raw_scores)
            return y - probs

    def compute_learner_weight(self, learner, X, y, pseudo_targets,
                                sample_weights, round_idx):
        if self.task_type == "regression":
            # Standard gradient boosting: alpha=1, shrinkage via learning_rate in ensemble
            return 1.0
        else:
            # For classification: use line search on log-loss
            preds = learner.predict(X)
            # Approximate optimal step size via Newton step
            probs = self._sigmoid(self._raw_scores)
            numerator = np.sum(pseudo_targets * preds)
            denominator = np.sum(probs * (1 - probs) * preds ** 2) + 1e-10
            alpha = numerator / denominator
            return max(alpha, 0.0)

    def update_weights(self, sample_weights, learner, X, y, pseudo_targets,
                       alpha, round_idx):
        # Gradient boosting doesn't reweight samples; it fits to pseudo-residuals.
        # But we update raw scores for classification gradient computation.
        if self.task_type == "classification":
            preds = learner.predict(X)
            self._raw_scores += self.learning_rate * alpha * preds
        # Weights stay uniform
        return sample_weights
"""

OPS = [
    {
        "op": "replace",
        "file": _FILE,
        "start_line": 147,
        "end_line": 256,
        "content": _CONTENT,
    },
]
