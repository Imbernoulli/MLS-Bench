# MLS-Bench: ml-anomaly-detection

# Unsupervised Anomaly Detection Algorithm Design

## Research Question
Design a novel unsupervised anomaly detection algorithm for tabular data that generalizes across datasets with different sample counts, dimensionality, and anomaly rates. The contribution is the *scoring rule* — how to model normal structure on standardized tabular features and assign higher scores to deviating points — using only unlabeled features at fit time.

## Background
Unsupervised anomaly detection identifies rare or unusual samples without labels during training. No single method dominates across dataset characteristics; promising designs combine density, isolation, distance, projection, ensemble, or robust-statistics ideas.

Reference baselines:
- **Isolation Forest (iForest)** — Liu, Ting, Zhou, ICDM 2008 ([paper](https://ieeexplore.ieee.org/document/4781136)). Tree-based isolation: anomalies are isolated with shorter random-partition path lengths. Default hyperparameters: 100 trees, sub-sample size 256.
- **Local Outlier Factor (LOF)** — Breunig, Kriegel, Ng, Sander, SIGMOD 2000. Density-based: ratio of a point's local reachability density to that of its k-nearest neighbors. Default `n_neighbors=20`.
- **One-Class SVM (OCSVM)** — Schölkopf, Platt, Shawe-Taylor, Smola, Williamson, 2001. Boundary-based: RBF kernel with `nu` controlling outlier fraction.
- **ECOD (Empirical Cumulative-distribution Outlier Detection)** — Li, Zhao, Hu, Botta, Ionescu, Chen, TKDE 2022 ([arXiv:2201.00382](https://arxiv.org/abs/2201.00382)). Per-dimension empirical CDFs; aggregate (negative) log tail probabilities across dimensions. Parameter-free.
- **COPOD (Copula-Based Outlier Detection)** — Li, Zhao, Botta, Ionescu, Hu, ICDM 2020 ([arXiv:2009.09463](https://arxiv.org/abs/2009.09463)). Empirical copula on per-dimension marginals; uses left/right/skewness-corrected tail probabilities. Parameter-free.

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

Available libraries: `numpy`, `scipy` (linear algebra, statistics, spatial, optimization), `scikit-learn` (PCA, KDE, NearestNeighbors, GaussianMixture, ...), `pyod` (IForest, LOF, OCSVM, ECOD, COPOD, KNN, HBOS, PCA, LODA, SUOD, ...).

## Fixed Pipeline
The training and evaluation pipeline (datasets, splitting, and metric computation) is fixed by the harness and not editable. The detector fits on unlabeled training features and produces anomaly scores on held-out test features; the contribution is the scoring rule only.


## Your Workspace

You are working inside `/workspace`. The package source tree
`/workspace/scikit-learn/` is the research scaffold for this task.

## Files You May Edit

You may **only** modify these files, and **only within the listed line ranges
(inclusive, 1-indexed)**. Edits that change code outside these ranges — or creating new files, or
deleting whole files — will cause your submission to be invalid.

The line numbers mark an editable **region**, not a fixed line-count budget: you
may add or remove lines inside it. Only code outside the editable ranges must
stay unchanged.

- `scikit-learn/custom_anomaly.py`
- editable lines **28–80**




## Readable Context


### `scikit-learn/custom_anomaly.py`  [EDITABLE — lines 28–80 only]

```python
     1: """Unsupervised Anomaly Detection Benchmark for MLS-Bench.
     2: 
     3: EDITABLE: CustomAnomalyDetector class -- the agent's anomaly detection algorithm.
     4: FIXED: input loading + prediction emit. The dataset identity, the train/test
     5: split, the test labels, and the metrics live in a host-only module the agent's
     6: process cannot import; this program loads a pre-generated standardized
     7: (train, test) pair, fits the detector unsupervised on the train split, and emits
     8: the test anomaly scores. The host-side parser regenerates the labels and scores
     9: AUROC + F1. Inputs are pre-standardized, exactly as before.
    10: """
    11: 
    12: import os
    13: import io
    14: import base64
    15: import warnings
    16: 
    17: import numpy as np
    18: from sklearn.base import BaseEstimator
    19: 
    20: warnings.filterwarnings("ignore")
    21: SEED = int(os.environ.get("SEED", "42"))
    22: np.random.seed(SEED)
    23: 
    24: 
    25: # =====================================================================
    26: # EDITABLE: Custom Anomaly Detector
    27: # =====================================================================
    28: class CustomAnomalyDetector:
    29:     """Custom unsupervised anomaly detection algorithm.
    30: 
    31:     You MUST implement:
    32:         - __init__(self): initialize any hyperparameters and internal state
    33:         - fit(self, X): train the detector on unlabeled data X (n_samples, n_features).
    34:                         This is UNSUPERVISED — you do not receive labels.
    35:         - decision_function(self, X): return anomaly scores for X.
    36:                         Shape: (n_samples,). Higher scores = more anomalous.
    37: 
    38:     Available libraries (pre-installed):
    39:         - numpy, scipy, scikit-learn (StandardScaler, PCA, KernelDensity, etc.)
    40:         - pyod (IForest, LOF, OCSVM, ECOD, COPOD, KNN, HBOS, PCA, LODA, etc.)
    41: 
    42:     The detector will be evaluated on tabular anomaly detection benchmarks via
    43:     a 60/40 stratified train/test split, measuring AUROC and F1.
    44: 
    45:     Design considerations:
    46:         - Anomalies are rare (typically 2-30% of data)
    47:         - Feature dimensions vary (6 to 36 features)
    48:         - Dataset sizes vary (1,800 to 49,000 samples)
    49:         - Data is pre-standardized before being passed to fit/decision_function
    50:         - Your algorithm should work WITHOUT labels (unsupervised)
    51:         - Consider: density estimation, distance-based, projection-based,
    52:           ensemble methods, or hybrid approaches
    53:     """
    54: 
    55:     def __init__(self):
    56:         """Initialize the anomaly detector."""
    57:         # Default: simple Isolation Forest wrapper
    58:         from pyod.models.iforest import IForest
    59: 
    60:         self.model = IForest(random_state=SEED)
    61: 
    62:     def fit(self, X):
    63:         """Fit the detector on unlabeled training data.
    64: 
    65:         Args:
    66:             X: numpy array of shape (n_samples, n_features), standardized
    67:         """
    68:         self.model.fit(X)
    69:         return self
    70: 
    71:     def decision_function(self, X):
    72:         """Compute anomaly scores for input data.
    73: 
    74:         Args:
    75:             X: numpy array of shape (n_samples, n_features), standardized
    76: 
    77:         Returns:
    78:             scores: numpy array of shape (n_samples,), higher = more anomalous
    79:         """
    80:         return self.model.decision_function(X)
    81: 
    82: 
    83: # =====================================================================
    84: # FIXED: input loading + prediction emit (do not modify below this line)
    85: # =====================================================================
    86: def _inputs_dir():
    87:     d = os.environ.get("ANOMALY_INPUTS_DIR")
    88:     if d:
    89:         return d
    90:     return os.path.join(os.path.dirname(os.path.abspath(__file__)), "_anomaly_inputs")
    91: 
    92: 
    93: def _load_input(env_name, seed):
    94:     path = os.path.join(_inputs_dir(), f"{env_name}_seed{seed}.npz.b64")
    95:     with open(path, "r") as f:
    96:         raw = base64.b64decode(f.read())
    97:     d = np.load(io.BytesIO(raw))
    98:     return d["X_train"], d["X_test"]
    99: 
   100: 
   101: def main():
   102:     env = os.environ.get("ENV", "")
   103:     if not env:
   104:         raise SystemExit("ENV not set")
   105:     seed = SEED
   106:     print(f"=== Anomaly detection benchmark: {env} (seed={seed}) ===", flush=True)
   107:     X_train, X_test = _load_input(env, seed)
   108:     print(f"Input: train={X_train.shape}, test={X_test.shape}", flush=True)
   109:     detector = CustomAnomalyDetector()
   110:     detector.fit(X_train)
   111:     scores = np.asarray(detector.decision_function(X_test), dtype=np.float64).ravel()
   112:     payload = base64.b64encode(np.ascontiguousarray(scores, dtype=np.float64).tobytes()).decode("ascii")
   113:     print(f"ANOMALY_PRED env={env} seed={seed} n={scores.shape[0]} scores={payload}", flush=True)
   114:     print("Done.", flush=True)
   115: 
   116: 
   117: if __name__ == "__main__":
   118:     main()
```

## Reference Baselines

The following are **read-only** reference implementations. Each shows what
the editable region of a strong baseline looks like, with a few lines of
surrounding context for orientation. Study them, but write your own
algorithm — repeating a baseline verbatim will be detected and scored as
a baseline reproduction.


### `isolation_forest` baseline — editable region  [READ-ONLY — reference implementation]

In `scikit-learn/custom_anomaly.py`:

```python
Lines 28–50:
    25: # =====================================================================
    26: # EDITABLE: Custom Anomaly Detector
    27: # =====================================================================
    28: class CustomAnomalyDetector:
    29:     """Isolation Forest anomaly detector.
    30: 
    31:     Ensemble of random isolation trees. Anomaly score is based on the
    32:     average path length to isolate each sample.
    33:     """
    34: 
    35:     def __init__(self):
    36:         from pyod.models.iforest import IForest
    37: 
    38:         self.model = IForest(
    39:             n_estimators=100,
    40:             max_samples="auto",
    41:             contamination=0.1,
    42:             random_state=SEED,
    43:         )
    44: 
    45:     def fit(self, X):
    46:         self.model.fit(X)
    47:         return self
    48: 
    49:     def decision_function(self, X):
    50:         return self.model.decision_function(X)
    51: 
    52: 
    53: 
```

### `lof` baseline — editable region  [READ-ONLY — reference implementation]

In `scikit-learn/custom_anomaly.py`:

```python
Lines 28–55:
    25: # =====================================================================
    26: # EDITABLE: Custom Anomaly Detector
    27: # =====================================================================
    28: class CustomAnomalyDetector:
    29:     """Local Outlier Factor anomaly detector (ADBench protocol).
    30: 
    31:     Applies MinMax normalization internally to match the preprocessing
    32:     used by ADBench (data_generator.py: MinMaxScaler().fit(X_train)).
    33:     LOF is density-based and extremely sensitive to feature scaling,
    34:     so this is required to reproduce the Table D4 numbers.
    35:     """
    36: 
    37:     def __init__(self):
    38:         from pyod.models.lof import LOF
    39: 
    40:         # PyOD defaults (matches ADBench with no hyperparameter tuning):
    41:         # n_neighbors=20, algorithm='auto', metric='minkowski', p=2,
    42:         # contamination=0.1.
    43:         self.model = LOF()
    44:         self._scaler = None
    45: 
    46:     def fit(self, X):
    47:         from sklearn.preprocessing import MinMaxScaler
    48:         self._scaler = MinMaxScaler()
    49:         Xs = self._scaler.fit_transform(X)
    50:         self.model.fit(Xs)
    51:         return self
    52: 
    53:     def decision_function(self, X):
    54:         Xs = self._scaler.transform(X)
    55:         return self.model.decision_function(Xs)
    56: 
    57: 
    58: 
```

### `ocsvm` baseline — editable region  [READ-ONLY — reference implementation]

In `scikit-learn/custom_anomaly.py`:

```python
Lines 28–52:
    25: # =====================================================================
    26: # EDITABLE: Custom Anomaly Detector
    27: # =====================================================================
    28: class CustomAnomalyDetector:
    29:     """One-Class SVM anomaly detector (ADBench protocol).
    30: 
    31:     Applies MinMax normalization internally to match ADBench's
    32:     preprocessing. Uses PyOD defaults: kernel='rbf', nu=0.5,
    33:     gamma='auto' (= 1/n_features).
    34:     """
    35: 
    36:     def __init__(self):
    37:         from pyod.models.ocsvm import OCSVM
    38: 
    39:         # PyOD default: kernel='rbf', nu=0.5, gamma='auto'.
    40:         self.model = OCSVM()
    41:         self._scaler = None
    42: 
    43:     def fit(self, X):
    44:         from sklearn.preprocessing import MinMaxScaler
    45:         self._scaler = MinMaxScaler()
    46:         Xs = self._scaler.fit_transform(X)
    47:         self.model.fit(Xs)
    48:         return self
    49: 
    50:     def decision_function(self, X):
    51:         Xs = self._scaler.transform(X)
    52:         return self.model.decision_function(Xs)
    53: 
    54: 
    55: 
```

### `ecod` baseline — editable region  [READ-ONLY — reference implementation]

In `scikit-learn/custom_anomaly.py`:

```python
Lines 28–41:
    25: # =====================================================================
    26: # EDITABLE: Custom Anomaly Detector
    27: # =====================================================================
    28: class CustomAnomalyDetector:
    29:     """ECOD anomaly detector (PyOD default, matches ADBench)."""
    30: 
    31:     def __init__(self):
    32:         from pyod.models.ecod import ECOD
    33: 
    34:         self.model = ECOD()
    35: 
    36:     def fit(self, X):
    37:         self.model.fit(X)
    38:         return self
    39: 
    40:     def decision_function(self, X):
    41:         return self.model.decision_function(X)
    42: 
    43: 
    44: 
```

### `copod` baseline — editable region  [READ-ONLY — reference implementation]

In `scikit-learn/custom_anomaly.py`:

```python
Lines 28–45:
    25: # =====================================================================
    26: # EDITABLE: Custom Anomaly Detector
    27: # =====================================================================
    28: class CustomAnomalyDetector:
    29:     """COPOD: Copula-Based Outlier Detection.
    30: 
    31:     Parameter-free method using empirical copula functions to model
    32:     the joint tail probability of observations across features.
    33:     """
    34: 
    35:     def __init__(self):
    36:         from pyod.models.copod import COPOD
    37: 
    38:         self.model = COPOD(contamination=0.1)
    39: 
    40:     def fit(self, X):
    41:         self.model.fit(X)
    42:         return self
    43: 
    44:     def decision_function(self, X):
    45:         return self.model.decision_function(X)
    46: 
    47: 
    48: 
```


## Tips

- Keep the function/class signatures of the editable regions identical;
  evaluation imports them by name.
- Determinism matters: seeds are fixed; don't introduce hidden randomness.
- The baseline implementations above are deliberately strong. Aim for an
  *algorithmic* improvement — many hyperparameters are locked outside the
  editable surface anyway.

Good luck.
