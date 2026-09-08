"""Group-wise temperature scaling baseline with James-Stein-style shrinkage.

Per-group temperatures are shrunk toward the global temperature in
log-space using alpha = n_g / (n_g + k_shrink). For small n_g, alpha is
near 0 and the group falls back to the global temperature; for large
n_g, alpha approaches 1 and the local fit dominates. This avoids the
small-subgroup overfitting that made unshrunk group_temperature_scaling
worse than plain temperature_scaling on shifted test sets.
"""

from pathlib import Path

_FILE = "scikit-learn/custom_subgroup_calibration.py"

OPS = [
    {
        "op": "replace",
        "file": _FILE,
        "start_line": 200,
        "end_line": 219,
        "content": """class CalibrationMethod:\n    \"\"\"Group temperature scaling with James-Stein shrinkage to global T.\"\"\"\n\n    def __init__(self):\n        self.eps = 1e-6\n        self.k_shrink = 200.0\n        self.group_temperatures_ = {}\n        self.global_temperature_ = 1.0\n\n    def _fit_temperature(self, probs, labels):\n        probs = np.asarray(probs).reshape(-1)\n        labels = np.asarray(labels).reshape(-1).astype(int)\n        logits = special.logit(np.clip(probs, self.eps, 1.0 - self.eps))\n\n        def objective(log_t):\n            t = float(np.exp(log_t))\n            cal = special.expit(logits / t)\n            p = np.clip(cal, self.eps, 1.0 - self.eps)\n            return float(-np.mean(labels * np.log(p) + (1 - labels) * np.log(1 - p)))\n\n        result = optimize.minimize_scalar(objective, bounds=(-3.0, 3.0), method=\"bounded\")\n        return float(np.exp(result.x)) if result.success else 1.0\n\n    def fit(self, probs, labels, groups=None):\n        probs = np.asarray(probs).reshape(-1)\n        labels = np.asarray(labels).reshape(-1).astype(int)\n        self.global_temperature_ = self._fit_temperature(probs, labels)\n        log_T_global = float(np.log(self.global_temperature_))\n        self.group_temperatures_ = {}\n        if groups is None:\n            return self\n        groups = np.asarray(groups).reshape(-1)\n        for g in np.unique(groups):\n            mask = groups == g\n            n_g = int(mask.sum())\n            if n_g < 20 or np.unique(labels[mask]).size < 2:\n                self.group_temperatures_[int(g)] = self.global_temperature_\n                continue\n            T_local = self._fit_temperature(probs[mask], labels[mask])\n            log_T_local = float(np.log(T_local))\n            alpha = n_g / (n_g + self.k_shrink)\n            log_T_g = alpha * log_T_local + (1.0 - alpha) * log_T_global\n            self.group_temperatures_[int(g)] = float(np.exp(log_T_g))\n        return self\n\n    def predict_proba(self, probs, groups=None):\n        probs = np.asarray(probs).reshape(-1)\n        logits = special.logit(np.clip(probs, self.eps, 1.0 - self.eps))\n        if groups is None:\n            temp = self.global_temperature_\n            return np.clip(special.expit(logits / temp), self.eps, 1.0 - self.eps)\n        groups = np.asarray(groups).reshape(-1)\n        out = np.empty_like(probs)\n        for g in np.unique(groups):\n            mask = groups == g\n            temp = self.group_temperatures_.get(int(g), self.global_temperature_)\n            out[mask] = special.expit(logits[mask] / temp)\n        return np.clip(out, self.eps, 1.0 - self.eps)\n""",
    }
]
