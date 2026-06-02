# MLS-Bench: ml-selective-deferral

# Selective Deferral Under Subgroup Shift

## Research Question
Design a selective-prediction / deferral policy for high-stakes tabular decisions. The base classifier and train/calibration/test pipeline are fixed; the contribution is the *acceptance rule* that decides — given a target coverage — which examples are accepted and which are deferred to a downstream reviewer or backup process.

A good policy should:
- keep selective risk low at the target coverage,
- avoid concentrating deferrals on one subgroup,
- preserve the ranking quality of its acceptance score as a confidence signal,
- be simple enough to fit and apply offline on modest compute.

## Background
Selective prediction lets a fixed classifier abstain when its prediction is unreliable. Under subgroup shift, naive confidence thresholds can defer disproportionately on one group while leaving others under-covered.

## Implementation Contract
Implement `SelectivePolicy` in `scikit-learn/custom_selective.py`:

```python
class SelectivePolicy:
    def __init__(self, target_coverage: float = TARGET_COVERAGE_DEFAULT,
                 random_state: int = 0):
        ...

    def fit(self, probs: np.ndarray, y_true: np.ndarray,
            groups: np.ndarray, X: np.ndarray | None = None) -> "SelectivePolicy":
        # probs: (n, n_classes) calibration-time base-model probabilities
        # y_true: (n,) calibration labels
        # groups: (n,) integer subgroup ids
        # X: optional raw features
        ...

    def acceptance_score(self, probs, groups, X=None) -> np.ndarray:
        # Higher score = more confident -> more likely to accept.
        ...

    def predict_accept(self, probs, groups, X=None) -> np.ndarray:
        # Boolean array: True = accept, False = defer.
        ...

    def calibration_summary(self) -> dict[str, float]:
        ...
```

You may implement a global threshold, a learned acceptance score, subgroup-specific thresholds, conformal mechanisms, or any compact policy fitting this interface. The base classifier and the train/calibration/test split are not editable.

## Your Workspace

You are working inside `/workspace`. The package source tree
`/workspace/scikit-learn/` is the research scaffold for this task.

## Files You May Edit

You may **only** modify these files, and **only within the listed line ranges
(inclusive, 1-indexed)**. Edits outside these ranges — or creating new files,
or deleting existing ones — are not permitted.

- `scikit-learn/custom_selective.py`
- editable lines **253–287**

## Readable Context

### `scikit-learn/custom_selective.py`  [EDITABLE — lines 253–287 only]

```python
     1: """Selective prediction / deferral scaffold.
     2:
     3: Fixed:
     4: - offline tabular dataset loading
     5: - train / calibration / test splits
     6: - base classifier training
     7: - metric computation
     8:
     9: Editable:
    10: - SelectivePolicy, which decides whether to accept or defer predictions
    11:   based on calibration outputs.
    12: """
    13:
    14: from __future__ import annotations
    15:
    16: import argparse
    17: import json
    18: import os
    19: import warnings
    20: from dataclasses import dataclass
    21: from pathlib import Path
    22: from typing import Callable
    23:
    24: import numpy as np
    25: import pandas as pd
    26: from sklearn.ensemble import GradientBoostingClassifier
    27: from sklearn.exceptions import ConvergenceWarning
    28: from sklearn.linear_model import LogisticRegression
    29: from sklearn.metrics import roc_auc_score
    30: from sklearn.model_selection import train_test_split
    31: from sklearn.pipeline import Pipeline
    32: from sklearn.preprocessing import StandardScaler
    33:
    34: warnings.filterwarnings("ignore", category=ConvergenceWarning)
    35:
    36: TARGET_COVERAGE_DEFAULT = 0.80
    37: DATA_HOME = os.environ.get("SKLEARN_DATA_HOME", "/data/sklearn")
    38:
    39:
    40: (... fixed data-loading and pipeline plumbing ...)
   248: # =============================================================================
   249: # EDITABLE REGION START
   250: # =============================================================================
   251:
   252:
   253: class SelectivePolicy:
   254:     """Policy that maps calibration outputs to accept / defer decisions.
   255:
   256:     The default implementation is intentionally conservative:
   257:     it accepts the top-confidence examples needed to reach the target coverage.
   258:     """
   259:
   260:
   261:     def __init__(self, target_coverage: float = TARGET_COVERAGE_DEFAULT, random_state: int = 0):
   262:         self.target_coverage = float(target_coverage)
   263:         self.random_state = int(random_state)
   264:         self.threshold_: float = 0.5
   265:         self.group_thresholds_: dict[int, float] = {}
   266:         self.meta_model_ = None
   267:         self.strategy_name = "global_threshold"
   268:
   269:     def fit(self, probs: np.ndarray, y_true: np.ndarray, groups: np.ndarray, X: np.ndarray | None = None) -> "SelectivePolicy":
   270:         scores = self.acceptance_score(probs, groups, X)
   271:         quantile = float(np.clip(1.0 - self.target_coverage, 0.0, 1.0))
   272:         self.threshold_ = float(np.quantile(scores, quantile))
   273:         self.group_thresholds_ = {}
   274:         self.meta_model_ = None
   275:         return self
   276:
   277:     def acceptance_score(self, probs: np.ndarray, groups: np.ndarray, X: np.ndarray | None = None) -> np.ndarray:
   278:         return np.max(probs, axis=1)
   279:
   280:     def predict_accept(self, probs: np.ndarray, groups: np.ndarray, X: np.ndarray | None = None) -> np.ndarray:
   281:         scores = self.acceptance_score(probs, groups, X)
   282:         return scores >= self.threshold_
   283:
   284:     def calibration_summary(self) -> dict[str, float]:
   285:         return {
   286:             "threshold": float(self.threshold_),
   287:         }
   288:
   289:
   290: # =============================================================================
   291: # EDITABLE REGION END
   292: # =============================================================================
```

## Tips

- Keep the function/class signatures of the editable regions identical.
- Determinism matters: seeds are fixed; don't introduce hidden randomness.

Good luck.
