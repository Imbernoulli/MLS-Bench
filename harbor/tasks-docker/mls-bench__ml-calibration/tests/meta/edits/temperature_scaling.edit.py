"""Temperature Scaling baseline (SOTA).

Single-parameter calibration: divides logits by a learned temperature T > 0.
Originally proposed for neural network calibration; here adapted for any
classifier by converting probabilities to log-odds first.

Reference: Guo et al., "On Calibration of Modern Neural Networks" (ICML 2017)
"""

_FILE = "scikit-learn/custom_calibration.py"

_CONTENT = """\
class CalibrationMethod(BaseEstimator):
    \"\"\"Temperature Scaling calibration.

    Learns a single temperature T that scales all logits: softmax(z/T).
    Optimized by minimizing NLL on the calibration set.
    \"\"\"

    def __init__(self):
        self.is_binary = None
        self.temperature_ = 1.0

    def fit(self, probs, labels):
        if probs.ndim == 1:
            self.is_binary = True
            # Convert to 2-class logits
            eps = 1e-15
            p = np.clip(probs, eps, 1 - eps)
            logits = np.column_stack([np.log(1 - p), np.log(p)])
        else:
            self.is_binary = False
            eps = 1e-15
            logits = np.log(np.clip(probs, eps, 1.0))

        def nll(T):
            T_val = max(T[0], 0.01)
            scaled = logits / T_val
            # Numerically stable softmax
            scaled = scaled - scaled.max(axis=1, keepdims=True)
            exp_scaled = np.exp(scaled)
            log_probs = scaled - np.log(exp_scaled.sum(axis=1, keepdims=True))
            return -log_probs[np.arange(len(labels)), labels.astype(int)].mean()

        result = optimize.minimize(nll, x0=[1.5], bounds=[(0.01, 20.0)],
                                   method="L-BFGS-B")
        self.temperature_ = max(result.x[0], 0.01)
        return self

    def predict_proba(self, probs):
        eps = 1e-15
        if self.is_binary:
            p = np.clip(probs, eps, 1 - eps)
            logits = np.column_stack([np.log(1 - p), np.log(p)])
        else:
            logits = np.log(np.clip(probs, eps, 1.0))

        scaled = logits / self.temperature_
        scaled = scaled - scaled.max(axis=1, keepdims=True)
        exp_scaled = np.exp(scaled)
        calibrated = exp_scaled / exp_scaled.sum(axis=1, keepdims=True)

        if self.is_binary:
            return calibrated[:, 1]
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
