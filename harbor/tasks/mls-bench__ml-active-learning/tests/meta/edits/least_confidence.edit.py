"""LeastConfidence (Uncertainty Sampling) baseline for ml-active-learning.

Reference: vendor/external_packages/badge/query_strategies/least_confidence.py
Paper: Lewis & Gale (1994), "A Sequential Algorithm for Training Text Classifiers"
Selects n samples where the model is least confident in its top prediction.
"""

_FILE = "badge/query_strategies/custom_sampling.py"

_CONTENT = """\
class CustomSampling(Strategy):
    \"\"\"Least Confidence (Uncertainty Sampling) — selects samples with lowest
    maximum predicted probability.\"\"\"

    def __init__(self, X, Y, idxs_lb, net, handler, args):
        super(CustomSampling, self).__init__(X, Y, idxs_lb, net, handler, args)

    def query(self, n):
        idxs_unlabeled = np.arange(self.n_pool)[~self.idxs_lb]
        probs = self.predict_prob(self.X[idxs_unlabeled], np.asarray(self.Y)[idxs_unlabeled])
        U = probs.max(1)[0]
        return idxs_unlabeled[U.sort()[1][:n]]
"""

OPS = [
    {
        "op": "replace",
        "file": _FILE,
        "start_line": 28,
        "end_line": 54,
        "content": _CONTENT,
    },
]
