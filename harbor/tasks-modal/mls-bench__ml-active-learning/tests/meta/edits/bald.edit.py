"""BALD (Bayesian Active Learning by Disagreement) baseline for ml-active-learning.

Reference: vendor/external_packages/badge/query_strategies/bayesian_active_learning_disagreement_dropout.py
Paper: Houlsby et al. (2011), "Bayesian Active Learning for Classification and
       Preference Learning"; Kirsch et al. (2019) "BatchBALD" (NeurIPS 2019)
Uses MC Dropout to estimate mutual information between predictions and model
parameters. Approximates BatchBALD via greedy BALD scoring.
"""

_FILE = "badge/query_strategies/custom_sampling.py"

_CONTENT = """\
class CustomSampling(Strategy):
    \"\"\"BALD — Bayesian Active Learning by Disagreement (MC Dropout).
    Selects samples where there is maximal disagreement across stochastic
    forward passes, approximating mutual information.\"\"\"

    def __init__(self, X, Y, idxs_lb, net, handler, args, n_drop=10):
        super(CustomSampling, self).__init__(X, Y, idxs_lb, net, handler, args)
        self.n_drop = n_drop

    def query(self, n):
        import torch
        idxs_unlabeled = np.arange(self.n_pool)[~self.idxs_lb]
        probs = self.predict_prob_dropout_split(
            self.X[idxs_unlabeled], self.Y.numpy()[idxs_unlabeled], self.n_drop
        )
        # Mean prediction across MC samples
        pb = probs.mean(0)
        # Total entropy: H[y | x, D]
        entropy1 = (-pb * torch.log(pb + 1e-10)).sum(1)
        # Expected entropy: E_theta[H[y | x, theta]]
        entropy2 = (-probs * torch.log(probs + 1e-10)).sum(2).mean(0)
        # BALD score = total entropy - expected entropy = mutual information
        U = entropy2 - entropy1
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
