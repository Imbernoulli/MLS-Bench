"""Histogram Binning baseline.

Partitions the probability space into equal-width bins and assigns each bin
the empirical fraction of positives. Simple and effective.

Reference: Zadrozny & Elkan, "Obtaining calibrated probability estimates from
decision trees and naive Bayesian classifiers" (ICML 2001)
"""

_FILE = "scikit-learn/custom_calibration.py"

_CONTENT = """\
class CalibrationMethod(BaseEstimator):
    \"\"\"Histogram Binning calibration.

    Divides [0,1] into equal-width bins, assigns each bin the empirical
    positive fraction from the calibration set.
    \"\"\"

    def __init__(self, n_bins=15):
        self.n_bins = n_bins
        self.is_binary = None
        self.bin_edges_ = None
        self.bin_values_ = None

    def fit(self, probs, labels):
        if probs.ndim == 1:
            self.is_binary = True
            self.bin_edges_, self.bin_values_ = self._fit_bins(probs, labels)
        else:
            self.is_binary = False
            n_classes = probs.shape[1]
            self.bin_edges_ = []
            self.bin_values_ = []
            for c in range(n_classes):
                binary_labels = (labels == c).astype(float)
                edges, values = self._fit_bins(probs[:, c], binary_labels)
                self.bin_edges_.append(edges)
                self.bin_values_.append(values)
        return self

    def _fit_bins(self, probs, labels):
        edges = np.linspace(0, 1, self.n_bins + 1)
        values = np.zeros(self.n_bins)
        for i in range(self.n_bins):
            lo, hi = edges[i], edges[i + 1]
            if i == self.n_bins - 1:
                mask = (probs >= lo) & (probs <= hi)
            else:
                mask = (probs >= lo) & (probs < hi)
            if mask.sum() > 0:
                values[i] = labels[mask].mean()
            else:
                values[i] = (lo + hi) / 2  # fallback
        return edges, values

    def _predict_bins(self, probs, edges, values):
        calibrated = np.zeros_like(probs)
        for i in range(self.n_bins):
            lo, hi = edges[i], edges[i + 1]
            if i == self.n_bins - 1:
                mask = (probs >= lo) & (probs <= hi)
            else:
                mask = (probs >= lo) & (probs < hi)
            calibrated[mask] = values[i]
        return np.clip(calibrated, 0, 1)

    def predict_proba(self, probs):
        if self.is_binary:
            return self._predict_bins(probs, self.bin_edges_, self.bin_values_)
        else:
            n_classes = probs.shape[1]
            calibrated = np.zeros_like(probs)
            for c in range(n_classes):
                calibrated[:, c] = self._predict_bins(
                    probs[:, c], self.bin_edges_[c], self.bin_values_[c]
                )
            calibrated = np.clip(calibrated, 1e-15, None)
            calibrated = calibrated / calibrated.sum(axis=1, keepdims=True)
            return calibrated
"""

OPS = [
    {
        "op": "replace",
        "file": _FILE,
        "start_line": 45,
        "end_line": 102,
        "content": _CONTENT,
    },
]
