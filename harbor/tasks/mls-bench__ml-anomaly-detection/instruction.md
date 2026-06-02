# MLS-Bench: ml-anomaly-detection

# Unsupervised Anomaly Detection Algorithm Design

## Research Question
Design a novel unsupervised anomaly detection algorithm for tabular data that
generalizes across datasets with different sample counts, dimensionality, and
anomaly rates. The contribution is the *scoring rule* — how to model normal
structure on standardized tabular features and assign higher scores to
deviating points — using only unlabeled features at fit time.

## Background
Unsupervised anomaly detection identifies rare or unusual samples without
labels during training. No single method dominates across dataset
characteristics; promising designs combine density, isolation, distance,
projection, ensemble, or robust-statistics ideas.

## Implementation Contract
Implement `CustomAnomalyDetector` in `custom_anomaly.py`:

```python
class CustomAnomalyDetector:
    def __init__(self):
        # Initialize hyperparameters and internal state
        ...

    def fit(self, X):
        # X: numpy array (n_samples, n_features), already standardized
        # (zero mean, unit variance). No labels used.
        return self

    def decision_function(self, X):
        # Return anomaly scores: numpy array (n_samples,)
        # Higher = more anomalous.
        return scores
```

Available libraries: `numpy`, `scipy` (linear algebra, statistics, spatial,
optimization), `scikit-learn` (PCA, KDE, NearestNeighbors, GaussianMixture,
...), `pyod` (various detector families).

## Your Workspace

You are working inside `/workspace`. The package source tree
`/workspace/scikit-learn/` is the research scaffold for this task.

## Files You May Edit

You may **only** modify these files, and **only within the listed line ranges
(inclusive, 1-indexed)**. Edits outside these ranges — or creating new files,
or deleting existing ones — will cause your submission to be rejected.

- `scikit-learn/custom_anomaly.py`
- editable lines **160–212**

## Readable Context

### `scikit-learn/custom_anomaly.py`  [EDITABLE — lines 160–212 only]

```python
   157: # =====================================================================
   158: # EDITABLE: Custom Anomaly Detector (lines 160-212)
   159: # =====================================================================
   160: class CustomAnomalyDetector:
   161:     """Custom unsupervised anomaly detection algorithm.
   162:
   163:     You MUST implement:
   164:         - __init__(self): initialize any hyperparameters and internal state
   165:         - fit(self, X): train the detector on unlabeled data X (n_samples, n_features).
   166:                         This is UNSUPERVISED — you do not receive labels.
   167:         - decision_function(self, X): return anomaly scores for X.
   168:                         Shape: (n_samples,). Higher scores = more anomalous.
   169:
   170:     Available libraries (pre-installed):
   171:         - numpy, scipy, scikit-learn (StandardScaler, PCA, KernelDensity, etc.)
   172:         - pyod (various detector families)
   173:
   174:     Design considerations:
   175:         - Anomalies are rare
   176:         - Feature dimensions and dataset sizes vary
   177:         - Data is pre-standardized before being passed to fit/decision_function
   178:         - Your algorithm should work WITHOUT labels (unsupervised)
   179:         - Consider: density estimation, distance-based, projection-based,
   180:           ensemble methods, or hybrid approaches
   181:     """
   182:
   183:     def __init__(self):
   184:         """Initialize the anomaly detector."""
   185:         from pyod.models.iforest import IForest
   186:
   187:         self.model = IForest(random_state=SEED)
   188:
   189:     def fit(self, X):
   190:         """Fit the detector on unlabeled training data.
   191:
   192:         Args:
   193:             X: numpy array of shape (n_samples, n_features), standardized
   194:         """
   195:         self.model.fit(X)
   196:         return self
   197:
   198:     def decision_function(self, X):
   199:         """Compute anomaly scores for input data.
   200:
   201:         Args:
   202:             X: numpy array of shape (n_samples, n_features), standardized
   203:
   204:         Returns:
   205:             scores: numpy array of shape (n_samples,), higher = more anomalous
   206:         """
   207:         return self.model.decision_function(X)
```

## Tips

- Keep the function/class signatures of the editable regions identical;
  they are imported by name.
- Determinism matters: seeds are fixed; don't introduce hidden randomness.
- Aim for an *algorithmic* contribution — many hyperparameters are locked
  outside the editable surface anyway.

Good luck.
