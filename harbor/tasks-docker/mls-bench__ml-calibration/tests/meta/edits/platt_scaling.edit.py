"""Platt Scaling baseline.

Fits a logistic regression (sigmoid) to map uncalibrated probabilities to
calibrated ones. For multiclass, applies per-class binary calibration
then renormalizes.

Reference: Platt, "Probabilistic Outputs for Support Vector Machines" (1999)
"""

_FILE = "scikit-learn/custom_calibration.py"

_CONTENT = """\
class CalibrationMethod(BaseEstimator):
    \"\"\"Platt Scaling (logistic/sigmoid calibration).

    Fits A*f + B through a sigmoid for each class, where f is the
    uncalibrated probability (log-odds transformed).
    \"\"\"

    def __init__(self):
        self.is_binary = None
        self.a_ = None
        self.b_ = None

    def fit(self, probs, labels):
        if probs.ndim == 1:
            self.is_binary = True
            self.a_, self.b_ = self._fit_sigmoid(probs, labels)
        else:
            self.is_binary = False
            n_classes = probs.shape[1]
            self.a_ = np.zeros(n_classes)
            self.b_ = np.zeros(n_classes)
            for c in range(n_classes):
                binary_labels = (labels == c).astype(float)
                self.a_[c], self.b_[c] = self._fit_sigmoid(probs[:, c], binary_labels)
        return self

    def _fit_sigmoid(self, probs, labels):
        \"\"\"Fit sigmoid parameters A, B: calibrated = 1 / (1 + exp(A*f + B)).\"\"\"
        # Transform to log-odds space, clip to avoid inf
        eps = 1e-12
        f = np.log(np.clip(probs, eps, 1 - eps) / np.clip(1 - probs, eps, 1 - eps))

        # Target probabilities (Platt's target encoding)
        n_pos = labels.sum()
        n_neg = len(labels) - n_pos
        t_pos = (n_pos + 1) / (n_pos + 2) if n_pos > 0 else 0.5
        t_neg = 1 / (n_neg + 2) if n_neg > 0 else 0.5
        target = np.where(labels > 0.5, t_pos, t_neg)

        def objective(params):
            a, b = params
            p = 1.0 / (1.0 + np.exp(a * f + b))
            p = np.clip(p, eps, 1 - eps)
            loss = -(target * np.log(p) + (1 - target) * np.log(1 - p)).mean()
            return loss

        result = optimize.minimize(objective, x0=[1.0, 0.0], method="L-BFGS-B")
        return result.x[0], result.x[1]

    def predict_proba(self, probs):
        eps = 1e-12
        if self.is_binary:
            f = np.log(np.clip(probs, eps, 1 - eps) / np.clip(1 - probs, eps, 1 - eps))
            calibrated = 1.0 / (1.0 + np.exp(self.a_ * f + self.b_))
            return np.clip(calibrated, 0, 1)
        else:
            n_classes = probs.shape[1]
            calibrated = np.zeros_like(probs)
            for c in range(n_classes):
                f = np.log(np.clip(probs[:, c], eps, 1 - eps) /
                           np.clip(1 - probs[:, c], eps, 1 - eps))
                calibrated[:, c] = 1.0 / (1.0 + np.exp(self.a_[c] * f + self.b_[c]))
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
