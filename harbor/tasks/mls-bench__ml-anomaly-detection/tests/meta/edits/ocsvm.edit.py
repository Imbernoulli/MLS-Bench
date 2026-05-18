"""One-Class SVM baseline for ml-anomaly-detection.

Reference: Scholkopf et al., "Estimating the Support of a High-Dimensional Distribution",
           Neural Computation 2001.
Implementation: pyod.models.ocsvm.OCSVM (scikit-learn backend).

One-Class SVM learns a decision boundary around normal data in a
kernel-induced feature space. Points outside the boundary are anomalies.

ADBench reference numbers (Table D4, NeurIPS 2022 Supplemental) use PyOD
defaults AND MinMax feature normalization (ADBench data_generator.py uses
MinMaxScaler before fitting). Our task scaffold pre-applies StandardScaler,
so to match the ADBench protocol we re-normalize to [0, 1] here. This is
critical for OCSVM because the RBF bandwidth depends on feature scale.
"""

_FILE = "scikit-learn/custom_anomaly.py"

_CONTENT = """\
class CustomAnomalyDetector:
    \"\"\"One-Class SVM anomaly detector (ADBench protocol).

    Applies MinMax normalization internally to match ADBench's
    preprocessing. Uses PyOD defaults: kernel='rbf', nu=0.5,
    gamma='auto' (= 1/n_features).
    \"\"\"

    def __init__(self):
        from pyod.models.ocsvm import OCSVM

        # PyOD default: kernel='rbf', nu=0.5, gamma='auto'.
        self.model = OCSVM()
        self._scaler = None

    def fit(self, X):
        from sklearn.preprocessing import MinMaxScaler
        self._scaler = MinMaxScaler()
        Xs = self._scaler.fit_transform(X)
        self.model.fit(Xs)
        return self

    def decision_function(self, X):
        Xs = self._scaler.transform(X)
        return self.model.decision_function(Xs)

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
