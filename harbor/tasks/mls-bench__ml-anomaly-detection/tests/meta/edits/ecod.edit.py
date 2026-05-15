"""ECOD baseline for ml-anomaly-detection.

Reference: Li et al., "ECOD: Unsupervised Outlier Detection Using Empirical
           Cumulative Distribution Functions", IEEE TKDE 2022.
Implementation: pyod.models.ecod.ECOD (default PyOD wrapper, as used by ADBench).

ECOD is a parameter-free, highly interpretable unsupervised outlier
detection method based on empirical cumulative distribution functions.
It computes per-dimension tail probabilities and aggregates them into
a single anomaly score. Outliers tend to lie in the tails of the
marginal distributions.

NOTE: PyOD's ECOD is transductive (decision_function concatenates train+test
before computing ECDFs). ADBench uses exactly this default behavior to
produce the reference numbers in Table D4 of the NeurIPS 2022 supplement,
so we match that protocol here.
"""

_FILE = "scikit-learn/custom_anomaly.py"

_CONTENT = """\
class CustomAnomalyDetector:
    \"\"\"ECOD anomaly detector (PyOD default, matches ADBench).\"\"\"

    def __init__(self):
        from pyod.models.ecod import ECOD

        self.model = ECOD()

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
