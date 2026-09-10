"""Isotonic Regression baseline.

Non-parametric calibration using isotonic (monotonically non-decreasing)
regression. For multiclass, applies per-class then renormalizes.

Reference: Zadrozny & Elkan, "Transforming Classifier Scores into Accurate
Multiclass Probability Estimates" (KDD 2002)
"""

_FILE = "scikit-learn/custom_calibration.py"

_CONTENT = """\
class CalibrationMethod(BaseEstimator):
    \"\"\"Isotonic Regression calibration.

    Fits a non-parametric, monotonically non-decreasing function
    from uncalibrated probabilities to calibrated ones.
    \"\"\"

    def __init__(self):
        self.is_binary = None
        self.calibrators_ = None

    def fit(self, probs, labels):
        from sklearn.isotonic import IsotonicRegression as IR

        if probs.ndim == 1:
            self.is_binary = True
            iso = IR(out_of_bounds="clip", y_min=0.0, y_max=1.0)
            iso.fit(probs, labels)
            self.calibrators_ = [iso]
        else:
            self.is_binary = False
            n_classes = probs.shape[1]
            self.calibrators_ = []
            for c in range(n_classes):
                binary_labels = (labels == c).astype(float)
                iso = IR(out_of_bounds="clip", y_min=0.0, y_max=1.0)
                iso.fit(probs[:, c], binary_labels)
                self.calibrators_.append(iso)
        return self

    def predict_proba(self, probs):
        if self.is_binary:
            calibrated = self.calibrators_[0].predict(probs)
            return np.clip(calibrated, 0, 1)
        else:
            n_classes = probs.shape[1]
            calibrated = np.zeros_like(probs)
            for c in range(n_classes):
                calibrated[:, c] = self.calibrators_[c].predict(probs[:, c])
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
