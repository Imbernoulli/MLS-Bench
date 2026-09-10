"""Temperature scaling baseline for subgroup calibration."""

from pathlib import Path

_FILE = "scikit-learn/custom_subgroup_calibration.py"

OPS = [
    {
        "op": "replace",
        "file": _FILE,
        "start_line": 200,
        "end_line": 219,
        "content": """class CalibrationMethod:\n    \"\"\"Global temperature scaling on positive-class probabilities.\"\"\"\n\n    def __init__(self):\n        self.eps = 1e-6\n        self.temperature_ = 1.0\n\n    def fit(self, probs, labels, groups=None):\n        probs = np.asarray(probs).reshape(-1)\n        labels = np.asarray(labels).reshape(-1).astype(int)\n        logits = special.logit(np.clip(probs, self.eps, 1.0 - self.eps))\n\n        def objective(log_t):\n            t = float(np.exp(log_t))\n            cal = special.expit(logits / t)\n            p = np.clip(cal, self.eps, 1.0 - self.eps)\n            return float(-np.mean(labels * np.log(p) + (1 - labels) * np.log(1 - p)))\n\n        result = optimize.minimize_scalar(objective, bounds=(-3.0, 3.0), method=\"bounded\")\n        self.temperature_ = float(np.exp(result.x)) if result.success else 1.0\n        return self\n\n    def predict_proba(self, probs, groups=None):\n        probs = np.asarray(probs).reshape(-1)\n        logits = special.logit(np.clip(probs, self.eps, 1.0 - self.eps))\n        return np.clip(special.expit(logits / self.temperature_), self.eps, 1.0 - self.eps)\n""",
    }
]
