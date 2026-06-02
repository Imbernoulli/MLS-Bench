# MLS-Bench: ml-calibration

# Probability Calibration Method Design

## Research Question
Design a novel post-hoc probability calibration method that maps a classifier's raw confidence estimates into well-calibrated probabilities. The base classifier and train/calibration/test splits are fixed; the contribution is the *calibration mapping itself*, learned only from a held-out calibration set.

## Background
A well-calibrated model satisfies: among all predictions where the model outputs probability p for a class, the empirical fraction that are correct is approximately p. Modern neural networks, GBMs, RFs, and SVMs are routinely miscalibrated.

## Implementation Contract
Implement `CalibrationMethod` in `custom_calibration.py`:

```python
class CalibrationMethod(BaseEstimator):
    def fit(self, probs, labels):
        # probs: (n,) for binary (positive-class probability)
        #        or (n, C) for multiclass (rows sum to 1)
        # labels: (n,) integer class labels
        return self

    def predict_proba(self, probs):
        # Returns calibrated probabilities of the same shape as input.
        return calibrated_probs
```

Available imports: `numpy`, `scipy` (`optimize`, `interpolate`, `special`), `sklearn`. The output must remain a valid probability distribution (non-negative, sums to 1 for multiclass).

## Your Workspace

You are working inside `/workspace`. The package source tree
`/workspace/scikit-learn/` is the research scaffold for this task.

## Files You May Edit

You may **only** modify these files, and **only within the listed line ranges
(inclusive, 1-indexed)**. Edits outside these ranges — or creating new files,
or deleting existing ones — will be rejected.

- `scikit-learn/custom_calibration.py`
- editable lines **45–102**

## Readable Context

### `scikit-learn/custom_calibration.py`  [EDITABLE — lines 45–102 only]

```python
     1: """Probability calibration scaffold.
     2:
     3: EDITABLE: CalibrationMethod class (fit + predict_proba).
     4: """
     5:
     6: import argparse
     7: import math
     8: import os
     9: import warnings
    10:
    11: import numpy as np
    12: from scipy import optimize, interpolate, special
    13: from sklearn.base import BaseEstimator
    14:
    15: warnings.filterwarnings("ignore")
    16:
    17:
    18: # ============================================================================
    19: # Calibration Method (EDITABLE)
    20: # ============================================================================
    21:
    22: # -- EDITABLE REGION START (lines 45-102) ------------------------------------
    45: class CalibrationMethod(BaseEstimator):
    46:     """Post-hoc probability calibration method.
    47:
    48:     Given a trained classifier's uncalibrated probability outputs, learn a
    49:     calibration mapping that produces well-calibrated probabilities.
    50:
    51:     For binary classification, fit() receives probabilities for the positive
    52:     class. For multiclass, it receives the full probability matrix.
    53:
    54:     Interface:
    55:         fit(probs, labels):
    56:             probs: np.ndarray, shape (n_samples,) for binary or
    57:                    (n_samples, n_classes) for multiclass.
    58:                    Uncalibrated probability outputs from a classifier
    59:                    on the calibration set.
    60:             labels: np.ndarray, shape (n_samples,) integer class labels.
    61:
    62:         predict_proba(probs) -> np.ndarray:
    63:             probs: same format as fit().
    64:             Returns calibrated probabilities, same shape as input.
    65:             For binary: 1-D array of positive-class probabilities in [0, 1].
    66:             For multiclass: 2-D array (n_samples, n_classes), rows sum to 1.
    67:     """
    68:
    69:     def __init__(self):
    70:         self.is_binary = None
    71:
    72:     def fit(self, probs, labels):
    73:         if probs.ndim == 1:
    74:             self.is_binary = True
    75:         else:
    76:             self.is_binary = False
    77:         return self
    78:
    79:     def predict_proba(self, probs):
    80:         if self.is_binary:
    81:             return np.clip(probs, 0, 1)
    82:         else:
    83:             probs = np.clip(probs, 1e-15, 1.0)
    84:             probs = probs / probs.sum(axis=1, keepdims=True)
    85:             return probs
   102: # -- EDITABLE REGION END (lines 45-102) --------------------------------------
```

## Tips

- Keep the function/class signatures of the editable regions identical.
- Determinism matters: seeds are fixed; don't introduce hidden randomness.

Good luck.
