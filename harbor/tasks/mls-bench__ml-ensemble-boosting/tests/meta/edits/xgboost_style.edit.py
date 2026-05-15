"""XGBoost-style (Chen & Guestrin, 2016) baseline for ml-ensemble-boosting.

XGBoost-inspired boosting: uses second-order (Newton) optimization with
gradient and Hessian of the loss, plus L2 regularization on leaf weights.

Reference: Chen, T. and Guestrin, C. "XGBoost: A Scalable Tree Boosting System."
KDD 2016.
"""

_FILE = "scikit-learn/custom_boosting.py"

_CONTENT = """\
class BoostingStrategy:
    \"\"\"XGBoost-style: second-order Newton boosting with regularization.\"\"\"

    def __init__(self, config):
        self.config = config
        self.task_type = config["task_type"]
        self.n_rounds = config["n_rounds"]
        self.learning_rate = config["learning_rate"]
        # L2 regularization on leaf weights (lambda in XGBoost)
        self.reg_lambda = 1.0
        # Track raw scores for gradient/Hessian computation
        self._raw_scores = None

    def init_weights(self, n_samples):
        self._raw_scores = np.zeros(n_samples)
        return np.ones(n_samples) / n_samples

    def _sigmoid(self, x):
        return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))

    def compute_targets(self, y, current_predictions, sample_weights, round_idx):
        if self.task_type == "regression":
            # Negative gradient of squared error = residuals
            return y - current_predictions
        else:
            # Negative gradient of log-loss
            probs = self._sigmoid(self._raw_scores)
            return y - probs

    def compute_learner_weight(self, learner, X, y, pseudo_targets,
                                sample_weights, round_idx):
        preds = learner.predict(X)
        if self.task_type == "regression":
            # Newton step: sum(gradient * pred) / (sum(hessian * pred^2) + lambda)
            # For squared error: gradient = residual, hessian = 1
            numerator = np.sum(pseudo_targets * preds)
            denominator = np.sum(preds ** 2) + self.reg_lambda
            alpha = numerator / denominator
            return max(alpha, 0.0)
        else:
            # For log-loss: hessian = p*(1-p)
            probs = self._sigmoid(self._raw_scores)
            hessians = probs * (1.0 - probs)
            numerator = np.sum(pseudo_targets * preds)
            denominator = np.sum(hessians * preds ** 2) + self.reg_lambda
            alpha = numerator / denominator
            return max(alpha, 0.0)

    def update_weights(self, sample_weights, learner, X, y, pseudo_targets,
                       alpha, round_idx):
        # XGBoost uses second-order info, not sample reweighting.
        # Update raw scores for next round's gradient computation.
        preds = learner.predict(X)
        self._raw_scores += self.learning_rate * alpha * preds
        # Weights stay uniform — boosting signal is in the pseudo-residuals
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
