# MLS-Bench: ml-active-learning

# Active Learning: Query Strategy Design

## Research Question
Design a novel pool-based active learning query strategy for tabular
classification. Strong strategies trade off uncertainty, diversity,
representativeness, and information gain. The fixed harness handles model
retraining and data management — the contribution is the *batch acquisition
rule itself*, not preprocessing or training-loop changes.

## Background
In pool-based active learning, a query strategy repeatedly selects a batch of
`n` examples from an unlabeled pool to be labeled by an oracle, then the model
is retrained on the expanded labeled set. The goal is to reach the highest
performance with the fewest labels.

## Implementation Contract
Modify `CustomSampling` in `badge/query_strategies/custom_sampling.py`:

```python
class CustomSampling(Strategy):
    def __init__(self, X, Y, idxs_lb, net, handler, args):
        super().__init__(X, Y, idxs_lb, net, handler, args)

    def query(self, n) -> np.ndarray:
        # Return n indices into self.X of currently-unlabeled samples to label.
        ...
```

Available from the `Strategy` base class:
- `self.X`, `self.Y`, `self.idxs_lb` — pool features (numpy `[n_pool, n_features]`), labels (LongTensor `[n_pool]`), boolean labeled mask.
- `self.n_pool` — total pool size.
- `self.predict_prob(X, Y)` — softmax probabilities `[len(X), n_classes]`.
- `self.predict_prob_dropout_split(X, Y, n_drop)` — MC dropout probabilities `[n_drop, len(X), n_classes]`.
- `self.get_embedding(X, Y)` — penultimate-layer embeddings `[len(X), emb_dim]`.
- `self.get_grad_embedding(X, Y)` — last-layer gradient embeddings `[len(X), emb_dim * n_classes]`.
- `self.get_exp_grad_embedding(X, Y)` — expected (per-class) Fisher embeddings `[len(X), n_classes, emb_dim]`.

## Your Workspace

You are working inside `/workspace`. The package source tree
`/workspace/badge/` is the research scaffold for this task.

## Files You May Edit

You may **only** modify these files, and **only within the listed line ranges
(inclusive, 1-indexed)**. Edits outside these ranges — or creating new files,
or deleting existing ones — will cause your submission to be rejected.

- `badge/query_strategies/custom_sampling.py`
- editable lines **28–54**

Other files you may **read** for context (do not modify):
- `badge/query_strategies/strategy.py`

## Readable Context

### `badge/query_strategies/custom_sampling.py`  [EDITABLE — lines 28–54 only]

```python
     1: """Custom active learning query strategy.
     2:
     3: This module defines a CustomSampling strategy that inherits from the
     4: framework's Strategy base class. The agent must implement the query() method
     5: to select the most informative samples from the unlabeled pool.
     6:
     7: Interface contract:
     8:   - self.X: numpy array of all pool features, shape (n_pool, n_features)
     9:   - self.Y: torch LongTensor of all pool labels, shape (n_pool,)
    10:   - self.idxs_lb: boolean array, True for labeled samples
    11:   - self.n_pool: total number of pool samples
    12:   - self.clf: the trained neural network model
    13:   - self.predict_prob(X, Y): returns softmax probabilities, shape (len(X), n_classes)
    14:   - self.predict_prob_dropout_split(X, Y, n_drop): returns MC dropout probs, shape (n_drop, len(X), n_classes)
    15:   - self.get_embedding(X, Y): returns penultimate-layer embeddings, shape (len(X), emb_dim)
    16:   - self.get_grad_embedding(X, Y): returns gradient embeddings, shape (len(X), emb_dim * n_classes)
    17:   - self.get_exp_grad_embedding(X, Y): returns expected Fisher embeddings, shape (len(X), n_classes, emb_dim)
    18:   - query(n) must return an array of n indices into self.X (indices of the UNLABELED pool)
    19: """
    20:
    21: import numpy as np
    22: from query_strategies.strategy import Strategy
    23:
    24:
    25: # ================================================================
    26: # EDITABLE REGION — Implement your query strategy below (lines 28-55)
    27: # ================================================================
    28: class CustomSampling(Strategy):
    29:     """Custom active learning query strategy.
    30:
    31:     Must implement query(n) -> np.ndarray of n indices from the unlabeled pool.
    32:     You may add helper methods, but query(n) is the entry point called by the
    33:     active learning loop.
    34:     """
    35:
    36:     def __init__(self, X, Y, idxs_lb, net, handler, args):
    37:         super(CustomSampling, self).__init__(X, Y, idxs_lb, net, handler, args)
    38:
    39:     def query(self, n):
    40:         """Select n samples from the unlabeled pool to label next.
    41:
    42:         Args:
    43:             n: number of samples to select
    44:
    45:         Returns:
    46:             np.ndarray of n indices (into self.X) of selected unlabeled samples
    47:         """
    48:         # Default: random sampling (replace with a better strategy)
    49:         idxs_unlabeled = np.arange(self.n_pool)[~self.idxs_lb]
    50:         return idxs_unlabeled[np.random.permutation(len(idxs_unlabeled))][:n]
    51:
    52: # ================================================================
    53: # END EDITABLE REGION
    54: # ================================================================
```

## Tips

- Keep the function/class signatures of the editable regions identical;
  they are imported by name.
- Determinism matters: seeds are fixed; don't introduce hidden randomness.
- Aim for an *algorithmic* contribution — many hyperparameters are locked
  outside the editable surface anyway.

Good luck.
