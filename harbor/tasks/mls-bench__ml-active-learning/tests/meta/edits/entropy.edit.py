"""EntropySampling baseline for ml-active-learning.

Reference: vendor/external_packages/badge/query_strategies/entropy_sampling.py
Paper: Shannon (1948); Settles (2009), "Active Learning Literature Survey"
Selects n samples with highest prediction entropy (maximum uncertainty).
"""

_FILE = "badge/query_strategies/custom_sampling.py"

_CONTENT = """\
class CustomSampling(Strategy):
    \"\"\"Entropy Sampling — selects samples with highest predictive entropy.\"\"\"

    def __init__(self, X, Y, idxs_lb, net, handler, args):
        super(CustomSampling, self).__init__(X, Y, idxs_lb, net, handler, args)

    def query(self, n):
        import torch
        idxs_unlabeled = np.arange(self.n_pool)[~self.idxs_lb]
        probs = self.predict_prob(self.X[idxs_unlabeled], self.Y.numpy()[idxs_unlabeled])
        log_probs = torch.log(probs)
        U = (probs * log_probs).sum(1)
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
