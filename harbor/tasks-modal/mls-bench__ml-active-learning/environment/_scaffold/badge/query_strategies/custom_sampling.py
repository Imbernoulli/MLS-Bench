"""Custom active learning query strategy.

This module defines a CustomSampling strategy that inherits from the badge
framework's Strategy base class. The agent must implement the query() method
to select the most informative samples from the unlabeled pool.

Interface contract:
  - self.X: numpy array of all pool features, shape (n_pool, n_features)
  - self.Y: torch LongTensor of all pool labels, shape (n_pool,)
  - self.idxs_lb: boolean array, True for labeled samples
  - self.n_pool: total number of pool samples
  - self.clf: the trained neural network model
  - self.predict_prob(X, Y): returns softmax probabilities, shape (len(X), n_classes)
  - self.predict_prob_dropout_split(X, Y, n_drop): returns MC dropout probs, shape (n_drop, len(X), n_classes)
  - self.get_embedding(X, Y): returns penultimate-layer embeddings, shape (len(X), emb_dim)
  - self.get_grad_embedding(X, Y): returns gradient embeddings (for BADGE), shape (len(X), emb_dim * n_classes)
  - self.get_exp_grad_embedding(X, Y): returns expected Fisher embeddings (for BAIT), shape (len(X), n_classes, emb_dim)
  - query(n) must return an array of n indices into self.X (indices of the UNLABELED pool)
"""

import numpy as np
from query_strategies.strategy import Strategy


# ================================================================
# EDITABLE REGION — Implement your query strategy below (lines 28-55)
# ================================================================
class CustomSampling(Strategy):
    """Custom active learning query strategy.

    Must implement query(n) -> np.ndarray of n indices from the unlabeled pool.
    You may add helper methods, but query(n) is the entry point called by the
    active learning loop.
    """

    def __init__(self, X, Y, idxs_lb, net, handler, args):
        super(CustomSampling, self).__init__(X, Y, idxs_lb, net, handler, args)

    def query(self, n):
        """Select n samples from the unlabeled pool to label next.

        Args:
            n: number of samples to select

        Returns:
            np.ndarray of n indices (into self.X) of selected unlabeled samples
        """
        # Default: random sampling (replace with a better strategy)
        idxs_unlabeled = np.arange(self.n_pool)[~self.idxs_lb]
        return idxs_unlabeled[np.random.permutation(len(idxs_unlabeled))][:n]

# ================================================================
# END EDITABLE REGION
# ================================================================
