"""Deep Isolation Forest (DIF) baseline for ml-anomaly-detection.

Reference: Xu et al., "Deep Isolation Forest for Anomaly Detection",
           IEEE TKDE 2023.
Implementation: pyod.models.dif.DIF.

Deep Isolation Forest extends Isolation Forest by using deep neural
network representations to perform isolation in learned feature spaces.
This enables it to handle complex, non-linear data distributions where
axis-aligned splits of vanilla Isolation Forest are insufficient.
"""

_FILE = "scikit-learn/custom_anomaly.py"

_CONTENT = """\
class CustomAnomalyDetector:
    \"\"\"DIF: Deep Isolation Forest.

    Extends Isolation Forest with neural network-learned representations
    for more effective isolation in complex feature spaces.
    \"\"\"

    def __init__(self):
        from pyod.models.dif import DIF

        self.model = DIF(
            max_samples=256,
            n_ensemble=50,
            contamination=0.1,
            random_state=SEED,
        )

    def fit(self, X):
        self.model.fit(X)
        return self

    def decision_function(self, X):
        return self.model.decision_function(X)

"""

OPS = [
    {
        "op": "replace",
        "file": _FILE,
        "start_line": 160,
        "end_line": 212,
        "content": _CONTENT,
    },
]
