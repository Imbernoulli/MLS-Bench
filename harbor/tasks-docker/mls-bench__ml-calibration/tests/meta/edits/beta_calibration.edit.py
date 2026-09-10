"""Beta Calibration baseline (SOTA).

Fits a beta calibration model: calibrated = 1/(1 + 1/exp(a*ln(p/(1-p)) + b*ln(1-p) + c)).
More flexible than Platt scaling with 3 parameters per class.

Reference: Kull et al., "Beta calibration: a well-founded and easily
implemented improvement on logistic calibration for binary classifiers"
(AISTATS 2017)
"""

_FILE = "scikit-learn/custom_calibration.py"

_CONTENT = """\
class CalibrationMethod(BaseEstimator):
    \"\"\"Beta Calibration.

    3-parameter model per class: uses log-odds and log(1-p) as features
    in a logistic regression, giving a richer calibration curve than Platt.
    \"\"\"

    def __init__(self):
        self.is_binary = None
        self.params_ = None

    def fit(self, probs, labels):
        if probs.ndim == 1:
            self.is_binary = True
            self.params_ = [self._fit_beta(probs, labels)]
        else:
            self.is_binary = False
            n_classes = probs.shape[1]
            self.params_ = []
            for c in range(n_classes):
                binary_labels = (labels == c).astype(float)
                self.params_.append(self._fit_beta(probs[:, c], binary_labels))
        return self

    def _fit_beta(self, probs, labels):
        \"\"\"Fit beta calibration: logit(q) = a*log(p/(1-p)) + b*log(1-p) + c.\"\"\"
        eps = 1e-12
        p = np.clip(probs, eps, 1 - eps)
        # Features: log(p/(1-p)) and log(1-p)
        f1 = np.log(p / (1 - p))  # log-odds
        f2 = np.log(1 - p)         # log(1-p)

        def objective(params):
            a, b, c = params
            logit_q = a * f1 + b * f2 + c
            q = 1.0 / (1.0 + np.exp(-logit_q))
            q = np.clip(q, eps, 1 - eps)
            loss = -(labels * np.log(q) + (1 - labels) * np.log(1 - q)).mean()
            return loss

        result = optimize.minimize(objective, x0=[1.0, 0.0, 0.0],
                                   method="L-BFGS-B")
        return result.x

    def _predict_beta(self, probs, params):
        eps = 1e-12
        p = np.clip(probs, eps, 1 - eps)
        a, b, c = params
        f1 = np.log(p / (1 - p))
        f2 = np.log(1 - p)
        logit_q = a * f1 + b * f2 + c
        q = 1.0 / (1.0 + np.exp(-logit_q))
        return np.clip(q, 0, 1)

    def predict_proba(self, probs):
        if self.is_binary:
            return self._predict_beta(probs, self.params_[0])
        else:
            n_classes = probs.shape[1]
            calibrated = np.zeros_like(probs)
            for c in range(n_classes):
                calibrated[:, c] = self._predict_beta(probs[:, c], self.params_[c])
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
