"""Isolation Forest baseline for ml-anomaly-detection.

Reference: Liu et al., "Isolation Forest", ICDM 2008.
Implementation: pyod.models.iforest.IForest (scikit-learn backend).

Isolation Forest detects anomalies by isolating observations via random
recursive partitioning. Anomalies require fewer splits to isolate, resulting
in shorter average path lengths in the ensemble of isolation trees.
"""

_FILE = "scikit-learn/custom_anomaly.py"

_CONTENT = """\
class CustomAnomalyDetector:
    \"\"\"Isolation Forest anomaly detector.

    Ensemble of random isolation trees. Anomaly score is based on the
    average path length to isolate each sample.
    \"\"\"

    def __init__(self):
        from pyod.models.iforest import IForest

        self.model = IForest(
            n_estimators=100,
            max_samples="auto",
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
