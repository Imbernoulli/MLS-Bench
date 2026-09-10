"""Isotonic regression baseline for subgroup calibration."""

from pathlib import Path

_FILE = "scikit-learn/custom_subgroup_calibration.py"

OPS = [
    {
        "op": "replace",
        "file": _FILE,
        "start_line": 200,
        "end_line": 219,
        "content": """class CalibrationMethod:\n    \"\"\"Isotonic regression calibration.\"\"\"\n\n    def __init__(self):\n        self.eps = 1e-6\n        self.model_ = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds=\"clip\")\n\n    def fit(self, probs, labels, groups=None):\n        probs = np.asarray(probs).reshape(-1)\n        labels = np.asarray(labels).reshape(-1).astype(int)\n        self.model_.fit(probs, labels)\n        return self\n\n    def predict_proba(self, probs, groups=None):\n        probs = np.asarray(probs).reshape(-1)\n        return np.clip(self.model_.predict(probs), self.eps, 1.0 - self.eps)\n""",
    }
]
