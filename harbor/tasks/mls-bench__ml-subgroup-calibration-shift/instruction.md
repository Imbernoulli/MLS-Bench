# MLS-Bench: ml-subgroup-calibration-shift

# Subgroup Calibration Under Distribution Shift

## Research Question
Design a post-hoc calibration method that remains reliable across subgroups when the test distribution shifts relative to calibration. The base tabular classifier and the (intentionally shifted) train/calibration/test splits are fixed; the contribution is the *calibration mapping* applied to positive-class probabilities, optionally using subgroup IDs.

## Background
Many calibration methods look accurate on average while remaining unreliable for protected or operationally meaningful subgroups, especially once subgroup prevalence or score distribution shifts at test time. The challenge is to improve worst-subgroup calibration without overfitting small per-group calibration samples.

## Implementation Contract
Modify `CalibrationMethod` in `scikit-learn/custom_subgroup_calibration.py`:

```python
class CalibrationMethod:
    def fit(self, probs, labels, groups=None):
        # probs: (n,) positive-class probabilities from the base classifier
        # labels: (n,) integer labels {0,1}
        # groups: (n,) integer subgroup IDs (may be None for group-agnostic methods)
        return self

    def predict_proba(self, probs, groups=None):
        # Returns (n,) calibrated positive-class probabilities in [0, 1].
        ...
```

The method must produce valid probabilities; `groups` may be ignored by group-agnostic methods.

## Your Workspace

You are working inside `/workspace`. The package source tree
`/workspace/scikit-learn/` is the research scaffold for this task.

## Files You May Edit

You may **only** modify these files, and **only within the listed line ranges
(inclusive, 1-indexed)**. Edits outside these ranges — or creating new files,
or deleting existing ones — are not permitted.

- `scikit-learn/custom_subgroup_calibration.py`
- editable lines **200–219**

## Readable Context

### `scikit-learn/custom_subgroup_calibration.py`  [EDITABLE — lines 200–219 only]

```python
     1: """Subgroup calibration under distribution shift.
     2:
     3: Fixed:
     4: - dataset loading
     5: - shifted train/calibration/test split
     6: - base classifier training
     7: - metric computation
     8:
     9: Editable:
    10: - CalibrationMethod
    11: """
    12:
    13: import argparse
    14: import os
    15: import warnings
    16:
    17: import numpy as np
    18: import pandas as pd
    19: from scipy import optimize, special
    20: from sklearn.isotonic import IsotonicRegression
    21: from sklearn.linear_model import LogisticRegression
    22: from sklearn.pipeline import Pipeline
    23: from sklearn.preprocessing import StandardScaler
    24:
    25: warnings.filterwarnings("ignore")
    26:
    27: DATA_HOME = os.environ.get("SKLEARN_DATA_HOME", "/data/sklearn")
    28:
    29: (... fixed dataset loading, shifted split, base-classifier training, metric helpers ...)
   199:
   200: class CalibrationMethod:
   201:     """Editable calibration method.
   202:
   203:     Implement fit() and predict_proba() to map raw positive-class probabilities
   204:     to calibrated positive-class probabilities.
   205:     """
   206:
   207:     def __init__(self):
   208:         self.eps = 1e-6
   209:         self._identity = True
   210:
   211:     def fit(self, probs, labels, groups=None):
   212:         probs = np.asarray(probs).reshape(-1)
   213:         labels = np.asarray(labels).reshape(-1).astype(int)
   214:         self._base_rate = float(np.clip(labels.mean(), self.eps, 1.0 - self.eps))
   215:         return self
   216:
   217:     def predict_proba(self, probs, groups=None):
   218:         probs = np.asarray(probs).reshape(-1)
   219:         return np.clip(probs, self.eps, 1.0 - self.eps)
```

## Tips

- Keep the function/class signatures of the editable regions identical.
- Determinism matters: seeds are fixed; don't introduce hidden randomness.

Good luck.
