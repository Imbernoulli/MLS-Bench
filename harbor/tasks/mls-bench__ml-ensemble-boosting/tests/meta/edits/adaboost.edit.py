"""AdaBoost (Freund & Schapire, 1997) baseline for ml-ensemble-boosting.

Classic adaptive boosting for classification: exponential loss reweighting.
For regression: AdaBoost.R2 with linear loss (Drucker, 1997).

Reference: Freund, Y. and Schapire, R.E. "A Decision-Theoretic Generalization
of On-Line Learning and an Application to Boosting." JCSS, 1997.
"""

_FILE = "scikit-learn/custom_boosting.py"

_CONTENT = """\
class BoostingStrategy:
    \"\"\"AdaBoost: exponential loss reweighting (classification) / AdaBoost.R2 (regression).\"\"\"

    def __init__(self, config):
        self.config = config
        self.task_type = config["task_type"]
        self.n_rounds = config["n_rounds"]
        self.learning_rate = config["learning_rate"]

    def init_weights(self, n_samples):
        return np.ones(n_samples) / n_samples

    def compute_targets(self, y, current_predictions, sample_weights, round_idx):
        if self.task_type == "classification":
            # AdaBoost fits on original labels (not residuals)
            return y
        else:
            # Regression: fit on negative gradient (residuals) so that the
            # fixed ensemble_predict accumulation (mean + sum alpha*lr*pred)
            # works correctly.
            return y - current_predictions

    def compute_learner_weight(self, learner, X, y, pseudo_targets,
                                sample_weights, round_idx):
        if self.task_type == "classification":
            preds = learner.predict(X)
            incorrect = (preds != y).astype(float)
            weighted_err = np.dot(sample_weights, incorrect) / sample_weights.sum()
            weighted_err = np.clip(weighted_err, 1e-10, 1.0 - 1e-10)
            alpha = self.learning_rate * 0.5 * np.log((1.0 - weighted_err) / weighted_err)
            return alpha
        else:
            # Regression: use alpha=1.0; shrinkage is applied by the fixed
            # ensemble_predict via learning_rate.  Sample reweighting in
            # update_weights handles the AdaBoost.R2 emphasis on hard examples.
            return 1.0

    def update_weights(self, sample_weights, learner, X, y, pseudo_targets,
                       alpha, round_idx):
        preds = learner.predict(X)
        if self.task_type == "classification":
            incorrect = (preds != y).astype(float)
            # w_i *= exp(alpha * I(wrong))
            sample_weights = sample_weights * np.exp(alpha * incorrect)
        else:
            # AdaBoost.R2-style: reduce weight on well-predicted samples
            # pseudo_targets are residuals; compare learner predictions to them
            errors = np.abs(preds - pseudo_targets)
            max_err = errors.max()
            if max_err > 0:
                errors = errors / max_err  # normalize to [0, 1]
            avg_loss = np.dot(sample_weights, errors)
            avg_loss = np.clip(avg_loss, 1e-10, 1.0 - 1e-10)
            beta = avg_loss / (1.0 - avg_loss)
            # Decrease weight for well-predicted samples
            sample_weights = sample_weights * np.power(beta, 1.0 - errors)
        # Normalize
        sample_weights = sample_weights / sample_weights.sum()
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
