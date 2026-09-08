"""Spline Calibration baseline (SOTA).

Uses cubic spline interpolation to learn a smooth, flexible calibration
mapping. The spline is fit on binned calibration data with natural boundary
conditions, providing a continuous non-parametric calibration curve.

Reference: Lucena, "Spline-Based Probability Calibration" (2018, arXiv:1809.07751)
Also: Gupta et al., "Calibration of Neural Networks using Splines" (ICLR 2021)
"""

_FILE = "scikit-learn/custom_calibration.py"

_CONTENT = """\
class CalibrationMethod(BaseEstimator):
    \"\"\"Spline Calibration.

    Fits a natural cubic spline on empirical calibration data (bin midpoints
    vs observed frequencies), producing a smooth calibration curve.
    Enforces monotonicity by post-processing.
    \"\"\"

    def __init__(self, n_knots=20):
        self.n_knots = n_knots
        self.is_binary = None
        self.splines_ = None

    def fit(self, probs, labels):
        if probs.ndim == 1:
            self.is_binary = True
            self.splines_ = [self._fit_spline(probs, labels)]
        else:
            self.is_binary = False
            n_classes = probs.shape[1]
            self.splines_ = []
            for c in range(n_classes):
                binary_labels = (labels == c).astype(float)
                self.splines_.append(self._fit_spline(probs[:, c], binary_labels))
        return self

    def _fit_spline(self, probs, labels):
        \"\"\"Fit a cubic spline mapping uncalibrated -> calibrated probs.\"\"\"
        # Create bins and compute empirical calibration
        sorted_idx = np.argsort(probs)
        sorted_probs = probs[sorted_idx]
        sorted_labels = labels[sorted_idx]

        # Use quantile-based bins for better coverage
        n_samples = len(probs)
        bin_size = max(n_samples // self.n_knots, 5)
        knot_x = []
        knot_y = []

        for i in range(0, n_samples, bin_size):
            end = min(i + bin_size, n_samples)
            if end - i < 3:
                continue
            knot_x.append(sorted_probs[i:end].mean())
            knot_y.append(sorted_labels[i:end].mean())

        if len(knot_x) < 3:
            # Fallback to identity
            return None

        knot_x = np.array(knot_x)
        knot_y = np.array(knot_y)

        # Add boundary knots
        if knot_x[0] > 0.01:
            knot_x = np.concatenate([[0.0], knot_x])
            knot_y = np.concatenate([[0.0], knot_y])
        if knot_x[-1] < 0.99:
            knot_x = np.concatenate([knot_x, [1.0]])
            knot_y = np.concatenate([knot_y, [1.0]])

        # Ensure monotonicity in knot values
        for i in range(1, len(knot_y)):
            knot_y[i] = max(knot_y[i], knot_y[i - 1])

        # Remove duplicate x values
        _, unique_idx = np.unique(knot_x, return_index=True)
        knot_x = knot_x[unique_idx]
        knot_y = knot_y[unique_idx]

        if len(knot_x) < 3:
            return None

        try:
            spline = interpolate.PchipInterpolator(knot_x, knot_y, extrapolate=True)
            return spline
        except Exception:
            return None

    def _predict_spline(self, probs, spline):
        if spline is None:
            return np.clip(probs, 0, 1)
        calibrated = spline(probs)
        return np.clip(calibrated, 0, 1)

    def predict_proba(self, probs):
        if self.is_binary:
            return self._predict_spline(probs, self.splines_[0])
        else:
            n_classes = probs.shape[1]
            calibrated = np.zeros_like(probs)
            for c in range(n_classes):
                calibrated[:, c] = self._predict_spline(probs[:, c], self.splines_[c])
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
