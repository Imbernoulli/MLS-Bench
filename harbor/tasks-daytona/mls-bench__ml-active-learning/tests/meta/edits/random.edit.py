"""RandomSampling baseline for ml-active-learning.

Reference: vendor/external_packages/badge/query_strategies/random_sampling.py
Selects n samples uniformly at random from the unlabeled pool.
"""

_FILE = "badge/query_strategies/custom_sampling.py"

_CONTENT = """\
class CustomSampling(Strategy):
    \"\"\"Random sampling baseline — selects samples uniformly at random.\"\"\"

    def __init__(self, X, Y, idxs_lb, net, handler, args):
        super(CustomSampling, self).__init__(X, Y, idxs_lb, net, handler, args)

    def query(self, n):
        idxs_unlabeled = np.arange(self.n_pool)[~self.idxs_lb]
        return idxs_unlabeled[np.random.permutation(len(idxs_unlabeled))][:n]
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
