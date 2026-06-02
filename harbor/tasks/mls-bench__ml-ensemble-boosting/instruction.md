# MLS-Bench: ml-ensemble-boosting

# Ensemble Boosting Strategy Design

## Research Question
Design a novel sample-weighting and update strategy for boosting that improves over standard methods across both classification and regression tasks. The contribution is the *strategy itself* (how sample weights are initialized and updated, what pseudo-targets each weak learner fits, how each learner is weighted), with shallow decision trees as the fixed weak learner.

## Background
Boosting builds an ensemble of weak learners sequentially, each round trying to correct errors left by previous rounds. Key design axes:
- **Pseudo-target computation**: original labels, negative gradients, or Newton-step targets using second-order information.
- **Learner weighting**: from weighted error, fixed with learning rate shrinkage, or via line search / Newton optimization.
- **Sample reweighting**: exponential reweighting of misclassified samples vs. uniform weights with pseudo-residual fitting.

## Implementation Contract
Modify `BoostingStrategy` in `scikit-learn/custom_boosting.py`:

```python
class BoostingStrategy:
    def init_weights(self, n_samples):
        # Initialize sample weights (should sum to 1).
        ...

    def compute_targets(self, y, current_predictions, sample_weights, round_idx):
        # Pseudo-targets the next weak learner will fit.
        ...

    def compute_learner_weight(self, learner, X, y, pseudo_targets,
                               sample_weights, round_idx):
        # Alpha for the just-fitted learner.
        ...

    def update_weights(self, sample_weights, learner, X, y,
                       pseudo_targets, alpha, round_idx):
        # Sample weights for the next round.
        ...
```

Available context: true labels, current ensemble predictions, sample weights, fitted learner (`learner.predict(X)`), round index, config dict with task metadata. Available imports in the FIXED section: `numpy`, `sklearn.tree`, `sklearn.metrics`, `sklearn.datasets`, `sklearn.model_selection`.

## Your Workspace

You are working inside `/workspace`. The package source tree
`/workspace/scikit-learn/` is the research scaffold for this task.

## Files You May Edit

You may **only** modify these files, and **only within the listed line ranges
(inclusive, 1-indexed)**. Edits outside these ranges — or creating new files,
or deleting existing ones — will be rejected.

- `scikit-learn/custom_boosting.py`
- editable lines **147–256**

## Readable Context

### `scikit-learn/custom_boosting.py`  [EDITABLE — lines 147–256 only]

```python
     1: """Boosting scaffold.
     2:
     3: EDITABLE: BoostingStrategy class.
     4: """
     5:
     6: import argparse
     7: import math
     8: import os
     9: import time
    10: from abc import ABC, abstractmethod
    11:
    12: import numpy as np
    13:
    14: # ============================================================================
    15: # EDITABLE — Boosting strategy (lines 147-256)
    16: # ============================================================================
    17:
   147: class BoostingStrategy:
   148:     """Sample weighting and update strategy for gradient boosting.
   149:
   150:     This class controls how sample weights are initialized, how pseudo-targets
   151:     (residuals or transformed targets) are computed for the next weak learner,
   152:     how learner weights (alphas) are determined, and how sample weights are
   153:     updated after each boosting round.
   154:
   155:     The strategy is used by the fixed training loop (below) which:
   156:     1. Calls init_weights() once at the start
   157:     2. For each round t = 0..T-1:
   158:        a. Calls compute_targets() to get pseudo-targets for fitting the learner
   159:        b. Fits a base learner on (X, pseudo_targets, sample_weights)
   160:        c. Calls compute_learner_weight() to get alpha_t
   161:        d. Calls update_weights() to adjust sample weights
   162:
   163:     Args (available via self.config set in __init__):
   164:         n_samples: int — number of training samples
   165:         n_features: int — number of input features
   166:         n_rounds: int — total boosting rounds
   167:         task_type: str — 'classification' or 'regression'
   168:         learning_rate: float — shrinkage factor
   169:
   170:     For classification: y in {0, 1}, use signed labels y_signed = 2*y - 1
   171:     For regression: y is continuous, use residual-based approaches
   172:     """
   173:
   174:     def __init__(self, config):
   175:         self.config = config
   176:         self.task_type = config["task_type"]
   177:         self.n_rounds = config["n_rounds"]
   178:         self.learning_rate = config["learning_rate"]
   179:
   180:     def init_weights(self, n_samples):
   181:         return np.ones(n_samples) / n_samples
   182:
   183:     def compute_targets(self, y, current_predictions, sample_weights, round_idx):
   184:         return y
   185:
   186:     def compute_learner_weight(self, learner, X, y, pseudo_targets,
   187:                                 sample_weights, round_idx):
   188:         return 1.0
   189:
   190:     def update_weights(self, sample_weights, learner, X, y, pseudo_targets,
   191:                        alpha, round_idx):
   192:         return sample_weights
   256:
```

## Tips

- Keep the function/class signatures of the editable regions identical.
- Determinism matters: seeds are fixed; don't introduce hidden randomness.

Good luck.
