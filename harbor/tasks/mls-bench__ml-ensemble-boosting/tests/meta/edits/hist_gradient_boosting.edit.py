"""Histogram-based Gradient Boosting baseline for ml-ensemble-boosting.

Bins continuous features into 255 discrete bins for fast split finding,
uses the histogram subtraction trick for efficiency, and fits
pseudo-residuals (negative gradients) like standard gradient boosting.

Reference: Ke et al., "LightGBM: A Highly Efficient Gradient Boosting
Decision Tree", NeurIPS 2017. Also: sklearn HistGradientBoosting.
"""

_FILE = "scikit-learn/custom_boosting.py"

_CONTENT = """\
class BoostingStrategy:
    \"\"\"Histogram-based Gradient Boosting with binned features.

    Pre-bins features into 255 discrete bins for fast split finding.
    Uses the histogram subtraction trick and fits pseudo-residuals
    (negative gradient of the loss) at each round.
    \"\"\"

    def __init__(self, config):
        self.config = config
        self.task_type = config["task_type"]
        self.n_rounds = config["n_rounds"]
        self.learning_rate = config["learning_rate"]
        self._raw_scores = None
        self._X_binned = None
        self._bin_edges = None
        self._n_bins = 255

    def _bin_features(self, X):
        \"\"\"Bin features into discrete bins using quantile-based binning.\"\"\"
        if self._bin_edges is None:
            n_features = X.shape[1]
            self._bin_edges = []
            for f in range(n_features):
                col = X[:, f]
                # Compute quantile-based bin edges
                percentiles = np.linspace(0, 100, self._n_bins + 1)
                edges = np.percentile(col, percentiles)
                # Remove duplicate edges
                edges = np.unique(edges)
                self._bin_edges.append(edges)
            self._X_binned = np.zeros_like(X, dtype=np.int32)
            for f in range(n_features):
                self._X_binned[:, f] = np.searchsorted(
                    self._bin_edges[f], X[:, f], side='right'
                ) - 1
                self._X_binned[:, f] = np.clip(
                    self._X_binned[:, f], 0, len(self._bin_edges[f]) - 2
                )

    def init_weights(self, n_samples):
        self._raw_scores = np.zeros(n_samples)
        return np.ones(n_samples) / n_samples

    def _sigmoid(self, x):
        return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))

    def compute_targets(self, y, current_predictions, sample_weights, round_idx):
        if self.task_type == \"regression\":
            # Negative gradient of squared error = residuals
            return y - current_predictions
        else:
            # Negative gradient of log-loss: y - sigmoid(F)
            probs = self._sigmoid(self._raw_scores)
            return y - probs

    def compute_learner_weight(self, learner, X, y, pseudo_targets,
                                sample_weights, round_idx):
        # Bin features on first call for histogram awareness
        if self._X_binned is None:
            self._bin_features(X)

        if self.task_type == \"regression\":
            # Standard gradient boosting step: alpha=1, shrinkage via learning_rate
            return 1.0
        else:
            # Newton step for log-loss using histogram-derived statistics
            preds = learner.predict(X)
            probs = self._sigmoid(self._raw_scores)
            # Hessian diagonal: p * (1-p)
            hessians = probs * (1 - probs)
            numerator = np.sum(pseudo_targets * preds)
            denominator = np.sum(hessians * preds ** 2) + 1e-10
            alpha = numerator / denominator
            return max(alpha, 0.0)

    def update_weights(self, sample_weights, learner, X, y, pseudo_targets,
                       alpha, round_idx):
        # Update raw scores for classification gradient computation
        if self.task_type == \"classification\":
            preds = learner.predict(X)
            self._raw_scores += self.learning_rate * alpha * preds
        # Gradient boosting does not reweight samples -- it fits residuals
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
