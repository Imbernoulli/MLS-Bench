"""Local Outlier Factor (LOF) baseline for ml-anomaly-detection.

Reference: Breunig et al., "LOF: Identifying Density-Based Local Outliers", SIGMOD 2000.
Implementation: pyod.models.lof.LOF (scikit-learn backend).

LOF measures the local deviation of density of a given sample with respect
to its neighbors. Samples with substantially lower density than their
neighbors are considered outliers.

ADBench reference numbers (Table D4, NeurIPS 2022 Supplemental) use PyOD
defaults AND MinMax feature normalization (ADBench data_generator.py uses
MinMaxScaler before fitting). Our task scaffold pre-applies StandardScaler,
so to match the ADBench protocol we re-normalize to [0, 1] here.
"""

_FILE = "scikit-learn/custom_anomaly.py"

_CONTENT = """\
class CustomAnomalyDetector:
    \"\"\"Local Outlier Factor anomaly detector (ADBench protocol).

    Applies MinMax normalization internally to match the preprocessing
    used by ADBench (data_generator.py: MinMaxScaler().fit(X_train)).
    LOF is density-based and extremely sensitive to feature scaling,
    so this is required to reproduce the Table D4 numbers.
    \"\"\"

    def __init__(self):
        from pyod.models.lof import LOF

        # PyOD defaults (matches ADBench with no hyperparameter tuning):
        # n_neighbors=20, algorithm='auto', metric='minkowski', p=2,
        # contamination=0.1.
        self.model = LOF()
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
